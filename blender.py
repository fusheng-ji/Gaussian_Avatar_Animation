import bpy
import math
from math import radians
from mathutils import Matrix
from mathutils import Vector
import json
import os
camera = bpy.context.scene.camera
camera.location = (0.0, 0.0, 100.0)
cam_data = camera.data
object_center = (0.0, 0.0, 0.0)
object_center = Vector(object_center)
# 旋转的次数和角度
num_rotations = 24
total_rotation_degrees = 360  # 完整一圈
rotation_step_degrees = total_rotation_degrees / num_rotations  # 每次旋转的角度

output_path = 'OUTPUT_PATH/blender/'
os.makedirs(output_path, exist_ok=True)
mask_dir = os.path.join(os.path.dirname(bpy.data.filepath), "mask")
os.makedirs(mask_dir, exist_ok=True)
scene = bpy.context.scene
# ================= 渲染器 & 全局光照（Cycles + 环境光） =================
# 切换到 Cycles，开启透明底（便于导出联合 mask）
scene.render.engine = 'CYCLES'
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True
scene.render.use_compositing = True  # 让合成节点参与渲染

# Cycles 关键参数（可按需求微调）
cycles = scene.cycles
cycles.samples = 128                # 总采样数（提高画质可加大）
cycles.use_adaptive_sampling = True
cycles.adaptive_min_samples = 0
cycles.time_limit = 0               # 不限时
cycles.max_bounces = 8              # 全局反弹次数
cycles.diffuse_bounces = 4
cycles.glossy_bounces = 4
cycles.transmission_bounces = 6
cycles.transparent_max_bounces = 8
cycles.volume_bounces = 2
cycles.sample_clamp_direct = 0.0
cycles.sample_clamp_indirect = 2.0  # 抑制间接高光火花
cycles.use_preview_denoising = True

# 开启视图层去噪（Final 渲染）
bpy.context.view_layer.cycles.use_denoising = True

# ===== 世界环境光：Sky (Nishita) 或 HDRI（若你有 HDRI 就填 hdri_path） =====
# 柔和环境光
world = scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    scene.world = world
world.use_nodes = True
wnodes = world.node_tree.nodes
wlinks = world.node_tree.links

# 清空原节点
wnodes.clear()

# 背景颜色节点
bg = wnodes.new(type="ShaderNodeBackground")
bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # 白色光，可改成其他颜色
bg.inputs["Strength"].default_value = 0.5  # 光照强度，0.5 比较柔和

# 输出节点
out = wnodes.new(type="ShaderNodeOutputWorld")

# 连接背景到输出
wlinks.new(bg.outputs["Background"], out.inputs["Surface"])


# 对准物体
def look_at(obj, target):
    
    direction = target - obj.location
    
    rot_quat = direction.to_track_quat('-Z', 'Z')
    
    obj.rotation_euler = rot_quat.to_euler()

look_at(camera, object_center)

# ================= Compositor：用 Alpha 生成联合二值 mask =================
scene.use_nodes = True
tree = scene.node_tree
nodes = tree.nodes
links = tree.links
nodes.clear()

# Render Layers
rl = nodes.new('CompositorNodeRLayers')

# Composite（保险：让合成链总是执行）
comp = nodes.new('CompositorNodeComposite')
links.new(rl.outputs['Image'], comp.inputs['Image'])

# 用 Alpha 做二值化：Alpha > 0.001 -> 1，否则 0
math_gt = nodes.new('CompositorNodeMath')
math_gt.operation = 'GREATER_THAN'
math_gt.inputs[1].default_value = 0.001  # 阈值，可按需要调
links.new(rl.outputs['Alpha'], math_gt.inputs[0])

# 输出到文件（单通道）
file_out = nodes.new('CompositorNodeOutputFile')
file_out.base_path = mask_dir
file_out.format.file_format = 'PNG'
file_out.format.color_mode = 'BW'
file_out.format.color_depth = '8'
file_out.file_slots[0].path = "image_######_mask"  # 用帧号填充
links.new(math_gt.outputs[0], file_out.inputs['Image'])

# ============== 采集相机参数 & 渲染 ============
camera_datas = []

def render(img_nm: str, frame_id: int):
    # 获取渲染尺寸
    scene.frame_set(frame_id)
    render = scene.render
    width = render.resolution_x
    height = render.resolution_y
    res_x = width
    res_y = height
    # 计算内参矩阵
    focal_length = cam_data.lens  # 相机焦距
    print('flcal_length : ',focal_length)
    sensor_width = cam_data.sensor_width  # 传感器宽度
    print('sensor_width',sensor_width)
    sensor_height = cam_data.sensor_height if cam_data.sensor_fit == 'VERTICAL' else sensor_width * height / width
    px = width / 2
    py = height / 2
    fx = width * focal_length / sensor_width
    fy = height * focal_length / sensor_height
    K = [
        [fx, 0, px],
        [0, fy, py],
        [0, 0, 1]
    ]

    # 计算sensor的宽度和高度
    if cam_data.sensor_fit == 'AUTO':
        if res_x > res_y:
            sensor_width = cam_data.sensor_width
            sensor_height = cam_data.sensor_width * res_y / res_x
        else:
            sensor_height = cam_data.sensor_height
            sensor_width = cam_data.sensor_height * res_x / res_y
    elif cam_data.sensor_fit == 'HORIZONTAL':
        sensor_width = cam_data.sensor_width
        sensor_height = cam_data.sensor_width * res_y / res_x
    else:  # VERTICAL
        sensor_height = cam_data.sensor_height
        sensor_width = cam_data.sensor_height * res_x / res_y

    # 计算fovx和fovy
    cam_lens = cam_data.lens
    fovx = 2 * math.atan(sensor_width / (2 * cam_lens)) * (180 / math.pi)  # 转换为度
    fovy = 2 * math.atan(sensor_height / (2 * cam_lens)) * (180 / math.pi)  # 转换为度

    world_to_camera_matrix = camera.matrix_world.inverted()
    print('world_to_camera_matrix',world_to_camera_matrix)
    # 获取外参矩阵（世界坐标系到相机坐标系的转换矩阵）
    c2w_matrix = camera.matrix_world
    print('c2w_matrix',c2w_matrix)
    c2w_list = []
    for row in c2w_matrix:
        r = [e for e in row]  # 将每一行中的元素转换为列表
        c2w_list.append(r)

    # 分解外参矩阵
    translation, rotation, scale = world_to_camera_matrix.decompose()
    # 转换为4x4矩阵
    rotation_matrix = rotation.to_matrix().to_4x4()
    translation_matrix = Matrix.Translation(translation)
    RT = translation_matrix @ rotation_matrix

    # 创建字典以保存相机参数
    camera_params = {
        'intrinsics': K,
        'extrinsics': {
            'c2w_matrix': c2w_list,
            'translation': translation[:],
            'rotation': rotation[:]
        },
        'width': width,
        'height': height,
        'fovx': fovx,
        'fovy': fovy,
        'img_id': img_nm
    }

    camera_datas.append(camera_params)



# 围绕Y轴均匀旋转24次
# 获取初始相机位置（已在开头设置）
initial_camera_position = Vector(camera.location) - object_center

for i in range(num_rotations):
    # 计算当前旋转角度
    current_angle_degrees = i * rotation_step_degrees
    current_angle_radians = math.radians(current_angle_degrees)
    
    # 创建旋转矩阵（从初始位置开始旋转）
    rotation_matrix = Matrix.Rotation(current_angle_radians, 4, 'Y')
    
    # 计算旋转后的相机位置
    rotated_position = rotation_matrix @ initial_camera_position
    
    # 设置相机位置
    camera.location = rotated_position + object_center
    
    # 让相机对准物体中心
    look_at(camera, object_center)
    
    # 更新场景
    bpy.context.view_layer.update()
    
    # 打印相机位置信息
    print(f"Rotation {i + 1}/{num_rotations}: angle={current_angle_degrees:.1f}°, position={camera.location}")
    
    # 渲染当前帧
    render_frame = str(i).zfill(6)
    img_nm = f'image_{render_frame}'
    render(img_nm, frame_id=i)
    bpy.context.scene.render.filepath = f'{output_path}image_{render_frame}'
    bpy.ops.render.render(write_still=True)

# 将字典转换为JSON字符串并保存到文件
with open('OUTPUT_PATH/cameras.json', 'w') as f:
    json.dump(camera_datas, f, indent=None)