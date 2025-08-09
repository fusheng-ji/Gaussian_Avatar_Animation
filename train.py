#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from avatar import Avatar, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

from pathlib import Path
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    wandb_run = prepare_output_and_logger(dataset, args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Avatar(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    for iteration in range(first_iter, opt.iterations + 1):        


        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:

            viewpoint_stack = scene.getTrainCameras().copy()
        idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand(3, device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.mask.unsqueeze(0).cuda()
        gt_image = gt_image * mask + bg[:, None, None] * (1. - mask)
        Ll1 = l1_loss(image, gt_image,mask)
        loss_ssim = 1.0 - ssim(image, gt_image)
        loss_ssim = loss_ssim * (mask.sum() / (image.shape[-1] * image.shape[-2]))
        bg_color_lpips = torch.rand_like(image)
        image_bg = image * mask + bg_color_lpips * (1. - mask)
        gt_image_bg = gt_image * mask + bg_color_lpips * (1. - mask)
        _, pred_patches, gt_patches = scene.patch_sampler.sample(mask, image_bg, gt_image_bg)
        loss_lpips = scene.lpips(pred_patches.clip(max=1), gt_patches).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (loss_ssim)+loss_lpips


        loss.backward()

        iter_end.record()

        with torch.no_grad():
            anim_path = './data/SFU/0005/0005_SideSkip001_poses.npz'
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                scene.render_canonical(iteration, nframes=60, is_train_progress=False, pose_type=None, pipe=pipe, bg=bg)
                scene.animate(anim_path, iteration, pipe=pipe, bg=background)

            # Log and save
            training_report(wandb_run, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, bg))
            if iteration % 3000 == 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                scene.render_canonical(iteration, nframes=60, is_train_progress=False, pose_type=None, pipe=pipe, bg=bg)
                scene.animate(anim_path, iteration, pipe=pipe, bg=background)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # Close wandb run
    if wandb_run:
        wandb.finish()

def prepare_output_and_logger(dataset, args):    
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create wandb logger
    wandb_run = None
    if WANDB_FOUND and not args.disable_wandb:
        # Generate time-based run name if not provided
        if args.wandb_name:
            run_name = args.wandb_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"gaussian_avatar_{timestamp}"
            
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=dataset.model_path,
            tags=args.wandb_tags if args.wandb_tags else None
        )
        print(f"Wandb logging initialized with run name: {run_name}")
    elif args.disable_wandb:
        print("Wandb logging disabled by user")
    else:
        print("Wandb not available: not logging progress")
    return wandb_run

def training_report(wandb_run, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, avatar : Avatar, renderFunc, renderArgs):
    if wandb_run:
        wandb.log({
            'train_loss_patches/l1_loss': Ll1.item(),
            'train_loss_patches/total_loss': loss.item(),
            'iter_time': elapsed
        }, step=iteration)

    # Report test and samples of training set
    #if iteration in testing_iterations:
    if iteration % 500 == 0 or iteration==1:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : avatar.getTestCameras()},
                              {'name': 'train', 'cameras' : [avatar.getTrainCameras()[idx % len(avatar.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, avatar.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_image.cuda()
                    mask = viewpoint.mask.unsqueeze(0).cuda()
                    gt_image = gt_image * mask + renderArgs[1][:, None, None] * (1. - mask)
                    gt_image = torch.clamp(gt_image, 0.0, 1.0)
                    if wandb_run and (idx < 5):
                        # Convert tensor to numpy for wandb logging
                        render_img = image.detach().cpu().numpy().transpose(1, 2, 0)
                        gt_img = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                        wandb.log({
                            f"{config['name']}_view_{viewpoint.image_name}/render": wandb.Image(render_img),
                            f"{config['name']}_view_{viewpoint.image_name}/ground_truth": wandb.Image(gt_img)
                        }, step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if wandb_run:
                    wandb.log({
                        f'{config["name"]}/loss_viewpoint - l1_loss': l1_test,
                        f'{config["name"]}/loss_viewpoint - psnr': psnr_test
                    }, step=iteration)

        if wandb_run:
            wandb.log({
                "scene/opacity_histogram": wandb.Histogram(avatar.gaussians.get_opacity.detach().cpu().numpy()),
                'total_points': avatar.gaussians.get_xyz.shape[0]
            }, step=iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Wandb related arguments
    parser.add_argument("--wandb_project", type=str, default="gaussian-avatar-animation", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_tags", nargs="+", type=str, default=[], help="Wandb tags for the run")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
