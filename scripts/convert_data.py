from isaacgym import gymapi # 假设此库可用
from myUtil import gpu_utl
import os

# --- 在所有其他 PyTorch 和 CUDA 相关导入之前设置环境变量 ---
try:
    available_gpus = gpu_utl.get_least_used_gpu()
    if available_gpus:
        TARGET_GPU_ID = available_gpus[-1] 
        print(f"myUtil.gpu_utl 检测到最空闲的GPU ID: {TARGET_GPU_ID}")
    else:
        print("警告: myUtil.gpu_utl 未返回可用GPU ID，将尝试默认行为 (GPU 0)。")
        TARGET_GPU_ID = "0" 

    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU_ID
    print(f"已设置 CUDA_VISIBLE_DEVICES='{TARGET_GPU_ID}'。")
    print(f"PyTorch 和相关库现在应该只能看到物理GPU ID {TARGET_GPU_ID} (并将其视为 cuda:0)。")
except ImportError:
    print("警告: 无法导入 'myUtil.gpu_utl'。将不会通过此工具选择GPU。")
    print("将尝试默认GPU或您之前设置的CUDA_VISIBLE_DEVICES（如果存在）。")
except Exception as e_gpu_select:
    print(f"警告: 使用 myUtil.gpu_utl 选择GPU时出错: {e_gpu_select}。将尝试默认行为。")
# -------------------------------------------------------------

import json
# import glob # 在您的代码中似乎没有用到 glob
import numpy as np
import cv2 # 主脚本中预处理图像也需要
from ultralytics import FastSAM # 主脚本中加载模型需要
import torch # 主脚本中设置DEVICE和预热需要
import time
import traceback
import open3d as o3d
import zarr 
from typing import Tuple, Dict # Union 在您的代码中似乎没有用到
from scipy.spatial.transform import Rotation as R_scipy
import shutil
# 从模型模块导入核心函数
from model.point_cloud_crop import create_point_cloud_from_depth_internal
from model.point_cloud_crop import save_point_cloud_to_html
from model.point_cloud_crop import segment_and_crop_with_fastsam as external_point_cloud_processor
import sys # 用于 sys.exit() (虽然在此版本中不再因单帧全零而退出)

# --- DEVICE 定义修正 ---
# 因为我们已经通过 CUDA_VISIBLE_DEVICES 限制了可见的 GPU，
# 如果 torch.cuda.is_available() 为 True，那么 "cuda:0" (或简称 "cuda") 就会指向我们选择的 GPU。
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0") # PyTorch会将CUDA_VISIBLE_DEVICES中的第一个GPU视为cuda:0
    print(f"PyTorch 检测到 CUDA 可用。")
    print(f"  将使用的PyTorch设备: {DEVICE}")
    try:
        print(f"  实际PyTorch设备名称: {torch.cuda.get_device_name(0)}")
        current_gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        current_gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"  当前选定GPU显存: 已分配 {current_gpu_memory_allocated:.2f} MiB, 已预留 {current_gpu_memory_reserved:.2f} MiB")
    except Exception as e_dev_name:
        print(f"  获取选定GPU名称或显存时出错: {e_dev_name}")
else:
    DEVICE = torch.device("cpu")
    print("PyTorch 未检测到 CUDA 可用，将使用 CPU。")
# --- DEVICE 定义结束 ---


NUM_POINTS_EXPECTED = 16384
OUTPUT_ZARR_PATH = '/data1/zhoufang/dataset_output/processed_data.zarr' # 您提供的路径
DATA_DIR = '/data1/zhoufang/dataset/collected_data'
MIN_FRAMES_PER_TRAJECTORY = 50 # 每条轨迹最少有效帧数
IMAGE_WIDTH_CONFIG = 640 # 用于数据加载和FastSAM的默认imgsz
IMAGE_HEIGHT_CONFIG = 480
FASTSAM_MODEL_PATH = './model/FastSAM-x.pt' # 确保路径正确
DEFAULT_TEXT_PROMPT_HAND = "robotic gripper with fingers"
DEFAULT_TEXT_PROMPT_OBJECT = "table"
DEPTH_SCALE = 1.0
print(f"重要提示: DEPTH_SCALE 设置为 {DEPTH_SCALE}。这假设您的原始深度数据单位是米。")
debug_first_frame_only = True # 根据您的代码设置为 False

# --- 相机与机器人姿态设置 (与之前相同) ---
cam_pos_cfg = [1, -0.8, 1.5]; cam_target_cfg = [0.5, 0.0, 0.5]
camera_p_world_gym = gymapi.Vec3(*cam_pos_cfg)
cam_forward = (gymapi.Vec3(*cam_target_cfg) - camera_p_world_gym).normalize()
cam_up_approx = gymapi.Vec3(0.0, 0.0, 1.0)
if abs(cam_forward.dot(cam_up_approx)) > 0.999: cam_up_approx = gymapi.Vec3(0.0, 1.0, 0.0)
cam_right = (cam_forward.cross(cam_up_approx)).normalize()
cam_up = (cam_right.cross(cam_forward)).normalize()
rotation_matrix_cols_np = np.array([[cam_right.x, cam_up.x, -cam_forward.x], [cam_right.y, cam_up.y, -cam_forward.y], [cam_right.z, cam_up.z, -cam_forward.z]], dtype=np.float64).T
camera_r_world_quat_gym = gymapi.Quat(0,0,0,1)
try:
    r_obj = R_scipy.from_matrix(rotation_matrix_cols_np); quat_xyzw_scipy = r_obj.as_quat()
    camera_r_world_quat_gym = gymapi.Quat(quat_xyzw_scipy[0], quat_xyzw_scipy[1], quat_xyzw_scipy[2], quat_xyzw_scipy[3])
except ValueError as e: print(f"错误: 创建四元数失败: {e}"); camera_r_world_quat_gym = gymapi.Quat(0,0,0,1)
CAMERA_WORLD_POSE_GYM = gymapi.Transform(p=camera_p_world_gym, r=camera_r_world_quat_gym)
robot_init_pos_cfg = [0.0, 0.0, 0.63]; robot_init_quat_cfg = [0.0, 0.0, 0.0, 1.0]
ROBOT_BASE_WORLD_POSE_GYM = gymapi.Transform(p=gymapi.Vec3(*robot_init_pos_cfg), r=gymapi.Quat(*robot_init_quat_cfg))

def gym_transform_to_matrix(transform: gymapi.Transform) -> np.ndarray:
    q = transform.r; p = transform.p
    r_matrix = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix = np.eye(4, dtype=np.float32); matrix[:3, :3] = r_matrix; matrix[:3, 3] = [p.x, p.y, p.z]
    return matrix

def convert_gymapi_quat_to_rpy_list(quat_obj: gymapi.Quat) -> list:
    euler_zyx = quat_obj.to_euler_zyx(); return [euler_zyx[2], euler_zyx[1], euler_zyx[0]]

def main():
    overall_start_time = time.time()
    print(f"将使用图像尺寸: Width={IMAGE_WIDTH_CONFIG}, Height={IMAGE_HEIGHT_CONFIG}")
    print(f"处理点云将使用设备: {DEVICE}") # DEVICE 现在是 torch.device 对象

    t_start_load_model = time.time()
    print(f"正在从 {FASTSAM_MODEL_PATH} 加载 FastSAM 模型...")
    if not os.path.exists(FASTSAM_MODEL_PATH):
        print(f"错误: FastSAM 模型未找到: {FASTSAM_MODEL_PATH}"); return
    try:
        fastsam_model_instance = FastSAM(FASTSAM_MODEL_PATH)
        print(f"FastSAM 模型已加载。耗时: {time.time() - t_start_load_model:.2f} 秒")
    except Exception as e:
        print(f"错误: 无法加载FastSAM模型: {e}"); traceback.print_exc(); return

    if DEVICE.type == 'cuda':
        print("  进行FastSAM模型预热 (用虚拟数据)...")
        t_warmup_start = time.time()
        try:
            dummy_img_warmup = np.zeros((IMAGE_HEIGHT_CONFIG, IMAGE_WIDTH_CONFIG, 3), dtype=np.uint8)
            _ = fastsam_model_instance.predict(source=dummy_img_warmup, texts=["a generic object"], device=DEVICE, retina_masks=True, imgsz=IMAGE_WIDTH_CONFIG, verbose=False, conf=0.1)
            torch.cuda.synchronize()
            print(f"  FastSAM模型预热完成。耗时: {time.time() - t_warmup_start:.2f} 秒")
        except Exception as e_warmup:
            print(f"  FastSAM模型预热时出错: {e_warmup}")
            print(f"  请检查选定GPU (通过 CUDA_VISIBLE_DEVICES 设置为 {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}) 是否有足够显存。")
    
    trajectory_files = []
    abs_data_dir = os.path.abspath(DATA_DIR)
    if not os.path.isdir(abs_data_dir):
        print(f"错误: 数据目录 '{abs_data_dir}' 不存在。"); return
    for root_dir, _, files_in_dir in os.walk(abs_data_dir):
        for filename in files_in_dir:
            if filename.startswith('trajectory_') and filename.endswith('.json'):
                trajectory_files.append(os.path.join(root_dir, filename))

    trajectory_files = sorted(trajectory_files)
    if not trajectory_files:
        print(f"错误：在'{abs_data_dir}'中没有找到轨迹文件"); return
    print(f"找到 {len(trajectory_files)} 条轨迹文件。")

    if trajectory_files and debug_first_frame_only:
        # ... (调试块代码保持不变，但内部计时已移除或减少) ...
        # 为了简洁，这里省略了调试块的完整代码，假设它与您之前的版本相似
        # 关键是确保它也使用正确的 DEVICE 对象，并且如果 external_point_cloud_processor 出错，
        # 它能提供有用的信息。
        print("[调试模式] 'debug_first_frame_only' 为 True，脚本将只处理第一帧并退出。")
        # (此处应有调试块的逻辑)
        # 假设调试块会调用 external_point_cloud_processor 并可能保存HTML等
        # ...
        return # 在调试模式下处理完后退出

    # --- Zarr 文件处理主循环 ---
    print(f"\n[主循环] 开始处理全部 {len(trajectory_files)} 条轨迹...")
    main_loop_start_time = time.time()

    if os.path.exists(OUTPUT_ZARR_PATH):
        print(f"警告: 输出文件 '{OUTPUT_ZARR_PATH}' 已存在。将删除并重新创建。")
        try:
            if os.path.isdir(OUTPUT_ZARR_PATH): shutil.rmtree(OUTPUT_ZARR_PATH)
            else: os.remove(OUTPUT_ZARR_PATH)
            print(f"旧的 '{OUTPUT_ZARR_PATH}' 已删除。")
        except Exception as e_del_zarr:
            print(f"错误: 删除旧的Zarr文件 '{OUTPUT_ZARR_PATH}' 失败: {e_del_zarr}"); return

    zarr_root = zarr.open_group(OUTPUT_ZARR_PATH, mode='a')
    data_group = zarr_root.require_group('data')
    meta_group = zarr_root.require_group('meta')
    
    episode_ends_list = []
    current_total_frames_in_zarr = 0
    pc_dataset, state_dataset, action_dataset = None, None, None
    first_trajectory_successfully_written = False

    main_loop_conf_hand = 0.25 
    main_loop_conf_object = 0.25
    print(f"[主循环] FastSAM置信度: 手={main_loop_conf_hand}, 物体={main_loop_conf_object}")
    main_loop_imgsz_w = IMAGE_WIDTH_CONFIG
    main_loop_imgsz_h = IMAGE_HEIGHT_CONFIG
    # print(f"[主循环] FastSAM输入尺寸: Width={main_loop_imgsz_w}, Height={main_loop_imgsz_h}")

    processed_traj_count = 0
    skipped_traj_count = 0 # 因各种原因（包括全零比例过高）而跳过的轨迹

    for i, traj_file_path in enumerate(trajectory_files):
        current_traj_dir = os.path.dirname(traj_file_path)
        try:
            with open(traj_file_path, 'r') as f_json: raw_trajectory_data = json.load(f_json)
            if not isinstance(raw_trajectory_data, list) or not raw_trajectory_data:
                skipped_traj_count += 1
                continue
        except Exception as e_load_json_main:
            print(f"    警告: 加载轨迹 {i+1} JSON ({os.path.basename(traj_file_path)}) 失败: {e_load_json_main}，跳过。")
            skipped_traj_count += 1
            continue
        
        current_traj_pc_list = []
        current_traj_states_list = []
        current_traj_actions_list = []
        
        initial_frames_skipped_in_traj = 0 # 因初步检查失败跳过的帧
        pcd_processing_attempted_in_traj = 0 # 实际尝试了点云处理的帧数
        zero_pcd_frames_in_traj = 0          # 点云处理后结果为全零的帧数
        # valid_frames_collected_in_traj 将通过 len(current_traj_pc_list) 得到

        for frame_idx, frame_data in enumerate(raw_trajectory_data):
            required_keys = {
                'depth_image_path': str, 'rgb_image_path': str, 'camera_K_matrix': list,
                'joint_positions': list, 'joint_velocities': list,
                'eef_target_position_relative_to_base': list,
                'eef_target_orientation_relative_to_base_xyzw': list,
                'gripper_openness_ratio': (float, int), 
                'object_position_relative_to_base': list,
                'object_orientation_relative_to_base_xyzw': list
            }
            valid_frame = True
            for key, exp_type in required_keys.items():
                val = frame_data.get(key)
                if val is None: valid_frame = False; break
                if isinstance(exp_type, tuple): 
                    if not any(isinstance(val, t) for t in exp_type): valid_frame = False; break
                elif not isinstance(val, exp_type): valid_frame = False; break
            if not valid_frame: 
                initial_frames_skipped_in_traj += 1
                continue
            
            k_matrix = np.array(frame_data['camera_K_matrix'], dtype=np.float32)
            if k_matrix.shape != (3,3): 
                initial_frames_skipped_in_traj += 1; continue
            
            depth_path = os.path.join(current_traj_dir, frame_data['depth_image_path'])
            rgb_path = os.path.join(current_traj_dir, frame_data['rgb_image_path'])
            try:
                depth_np = np.load(depth_path); rgb_np_loaded = np.load(rgb_path)
                if rgb_np_loaded.shape[:2] != (IMAGE_HEIGHT_CONFIG, IMAGE_WIDTH_CONFIG) or \
                   depth_np.shape[:2] != (IMAGE_HEIGHT_CONFIG, IMAGE_WIDTH_CONFIG):
                    initial_frames_skipped_in_traj += 1; continue
                if not np.issubdtype(depth_np.dtype, np.floating): depth_np = depth_np.astype(np.float32)
            except FileNotFoundError: initial_frames_skipped_in_traj += 1; continue
            except Exception: initial_frames_skipped_in_traj += 1; continue
            
            rgb_input_main_uint8 = rgb_np_loaded.copy()
            if not np.issubdtype(rgb_input_main_uint8.dtype, np.uint8):
                if np.issubdtype(rgb_input_main_uint8.dtype, np.floating) and rgb_input_main_uint8.max()<=1.0 and rgb_input_main_uint8.min()>=0.0:
                    rgb_input_main_uint8=(rgb_input_main_uint8*255).astype(np.uint8)
                else: rgb_input_main_uint8=np.clip(rgb_input_main_uint8,0,255).astype(np.uint8)

            if rgb_input_main_uint8.ndim == 3 and rgb_input_main_uint8.shape[2] == 4: 
                rgb_bgr_main = cv2.cvtColor(rgb_input_main_uint8, cv2.COLOR_RGBA2BGR)
            elif rgb_input_main_uint8.ndim == 3 and rgb_input_main_uint8.shape[2] == 3:
                rgb_bgr_main = cv2.cvtColor(rgb_input_main_uint8, cv2.COLOR_RGB2BGR)
            elif rgb_input_main_uint8.ndim == 2: 
                rgb_bgr_main = cv2.cvtColor(rgb_input_main_uint8, cv2.COLOR_GRAY2BGR)
            else: 
                initial_frames_skipped_in_traj += 1; continue
            
            # --- 增加尝试处理计数器 ---
            pcd_processing_attempted_in_traj += 1
            # --- 计数器结束 ---

            sampled_points_np = np.zeros((NUM_POINTS_EXPECTED, 6), dtype=np.float32)
            
            try:
                print(f"    轨迹 {i+1}, 帧 {frame_idx}: 调用 external_point_cloud_processor...")
                processed_o3d_pcd = external_point_cloud_processor(
                    rgb_image_input=rgb_bgr_main,
                    depth_image_input=depth_np, camera_k_matrix=k_matrix,
                    depth_processing_scale=DEPTH_SCALE,
                    fastsam_model_instance=fastsam_model_instance,
                    text_prompt_hand=DEFAULT_TEXT_PROMPT_HAND,
                    text_prompt_object=DEFAULT_TEXT_PROMPT_OBJECT,
                    device_for_fastsam=DEVICE, 
                    return_separate_segments=False, 
                    image_width_config_for_fastsam=main_loop_imgsz_w,
                    image_height_config_for_fastsam=main_loop_imgsz_h,
                    ground_removal_z_threshold=None, 
                    return_debug_images=False,
                    fastsam_conf_hand=main_loop_conf_hand,
                    fastsam_conf_object=main_loop_conf_object,
                    timing_prefix="" # 确保传递，以匹配您 model 模块中的 dbg_prefix
                )
                
                # --- 新增 DEBUG 打印 ---
                if processed_o3d_pcd is None:
                    print(f"    轨迹 {i+1}, 帧 {frame_idx}: external_point_cloud_processor 返回了 None。")
                else:
                    print(f"    轨迹 {i+1}, 帧 {frame_idx}: external_point_cloud_processor 返回类型: {type(processed_o3d_pcd)}")
                    if isinstance(processed_o3d_pcd, o3d.geometry.PointCloud):
                        print(f"    轨迹 {i+1}, 帧 {frame_idx}: processed_o3d_pcd.has_points() = {processed_o3d_pcd.has_points()}")
                        if processed_o3d_pcd.has_points():
                            print(f"    轨迹 {i+1}, 帧 {frame_idx}: processed_o3d_pcd 点数 = {len(processed_o3d_pcd.points)}")
                            if len(processed_o3d_pcd.points) > 0:
                                print(f"    轨迹 {i+1}, 帧 {frame_idx}: processed_o3d_pcd 第一个点 (前3个坐标): {np.asarray(processed_o3d_pcd.points)[0, :3]}")
                        else:
                            print(f"    轨迹 {i+1}, 帧 {frame_idx}: processed_o3d_pcd 没有点。")
                    else:
                        print(f"    轨迹 {i+1}, 帧 {frame_idx}: processed_o3d_pcd 不是 Open3D PointCloud 对象。")
                # --- DEBUG 打印结束 ---
                
                if processed_o3d_pcd and processed_o3d_pcd.has_points():
                    print(f"    轨迹 {i+1}, 帧 {frame_idx}: 进入 sampled_points_np 生成逻辑。")
                    points_xyz = np.asarray(processed_o3d_pcd.points, dtype=np.float32)
                    print(f"    轨迹 {i+1}, 帧 {frame_idx}: points_xyz.shape = {points_xyz.shape}")
                    
                    if processed_o3d_pcd.has_colors(): 
                        points_rgb = np.asarray(processed_o3d_pcd.colors, dtype=np.float32)
                        print(f"    轨迹 {i+1}, 帧 {frame_idx}: points_rgb.shape (从颜色获取) = {points_rgb.shape}")
                    else: 
                        points_rgb = np.full((points_xyz.shape[0], 3), 0.5, dtype=np.float32)
                        print(f"    轨迹 {i+1}, 帧 {frame_idx}: points_rgb.shape (默认灰色) = {points_rgb.shape}")
                    
                    if points_xyz.shape[0] > 0: # 这个条件理论上已由 processed_o3d_pcd.has_points() 保证
                        points_xyzrgb = np.hstack((points_xyz, points_rgb))
                        print(f"    轨迹 {i+1}, 帧 {frame_idx}: points_xyzrgb.shape = {points_xyzrgb.shape}")
                        num_curr = points_xyzrgb.shape[0]
                        
                        if num_curr > NUM_POINTS_EXPECTED:
                            print(f"    轨迹 {i+1}, 帧 {frame_idx}: 点数 ({num_curr}) > 期望点数 ({NUM_POINTS_EXPECTED})，进行下采样。")
                            indices = np.random.choice(num_curr, NUM_POINTS_EXPECTED, replace=False)
                            sampled_points_np = points_xyzrgb[indices]
                        elif num_curr > 0: # num_curr < NUM_POINTS_EXPECTED and num_curr > 0
                            print(f"    轨迹 {i+1}, 帧 {frame_idx}: 点数 ({num_curr}) < 期望点数 ({NUM_POINTS_EXPECTED})，进行填充。")
                            padding_indices = np.random.choice(num_curr, NUM_POINTS_EXPECTED - num_curr, replace=True)
                            sampled_points_np = np.vstack((points_xyzrgb, points_xyzrgb[padding_indices]))
                        # 如果 num_curr == 0 (理论上不应在此分支内发生), sampled_points_np 保持为全零
                        print(f"    轨迹 {i+1}, 帧 {frame_idx}: sampled_points_np 赋值后 shape: {sampled_points_np.shape}")
                    else:
                        print(f"    轨迹 {i+1}, 帧 {frame_idx}: points_xyz.shape[0] 为 0，sampled_points_np 将保持全零。")
                else:
                    print(f"    轨迹 {i+1}, 帧 {frame_idx}: processed_o3d_pcd 为 None 或没有点，sampled_points_np 将保持全零。")

            except Exception as e_pcd_proc_main_loop:
                print(f"    !!!!!! 轨迹 {i+1}, 帧 {frame_idx}: 在 sampled_points_np 生成逻辑中发生异常: {e_pcd_proc_main_loop} !!!!!!")
                traceback.print_exc() # 打印详细的异常堆栈
                # sampled_points_np 保持为全零
                pass 
            # --- 修改：点云检查 ---
            if pcd_processing_attempted_in_traj > 0: # 确保是在尝试处理点云之后
                pcd_min = np.min(sampled_points_np)
                pcd_max = np.max(sampled_points_np)
                pcd_mean = np.mean(sampled_points_np)
                # 打印统计信息，帮助诊断
                print(f"    轨迹 {i+1}, 帧 {frame_idx}: sampled_points_np 统计: Min={pcd_min:.4e}, Max={pcd_max:.4e}, Mean={pcd_mean:.4e}")

                # 使用一个小的容差来判断是否“几乎”全为零
                is_effectively_zero = np.all(np.abs(sampled_points_np) < 1e-9) 

                if is_effectively_zero:
                    print(f"    警告: 轨迹 {i+1}, 帧 {frame_idx}: 点云数据被判定为全零或几乎全零 (基于容差检查)，跳过此帧。")
                    zero_pcd_frames_in_traj += 1
                    continue # 跳过此帧的后续处理和添加
                else:
                    # 点云不被认为是全零，可以进行保存
                    print(f"    轨迹 {i+1}, 帧 {frame_idx}: 点云包含有效数据。即将保存的点云数量 (经采样/填充后): {sampled_points_np.shape[0]}")
                    # 只有在点云有效时才添加到列表并处理后续
                    try:
                        # --- 开始定义 state_vector 及其组件 ---
                        print(f"      轨迹 {i+1}, 帧 {frame_idx}: 正在定义 state_vector 组件...")
                        
                        if 'joint_positions' not in frame_data: raise KeyError("'joint_positions' not in frame_data")
                        joint_pos_np = np.array(frame_data['joint_positions'], dtype=np.float32)
                        print(f"        joint_pos_np shape: {joint_pos_np.shape}")

                        if 'joint_velocities' not in frame_data: raise KeyError("'joint_velocities' not in frame_data")
                        joint_vel_np = np.array(frame_data['joint_velocities'], dtype=np.float32)
                        print(f"        joint_vel_np shape: {joint_vel_np.shape}")

                        if 'gripper_openness_ratio' not in frame_data: raise KeyError("'gripper_openness_ratio' not in frame_data")
                        gripper_state_np = np.array([frame_data['gripper_openness_ratio']], dtype=np.float32)
                        print(f"        gripper_state_np shape: {gripper_state_np.shape}")

                        # 对于可能缺失的与物体相关的键，进行更严格的检查
                        if 'object_position_relative_to_base' not in frame_data: raise KeyError("'object_position_relative_to_base' not in frame_data")
                        obj_pos_rel_list = frame_data['object_position_relative_to_base']
                        print(f"        obj_pos_rel_list: {obj_pos_rel_list}")

                        if 'object_orientation_relative_to_base_xyzw' not in frame_data: raise KeyError("'object_orientation_relative_to_base_xyzw' not in frame_data")
                        obj_ori_rel_list_xyzw = frame_data['object_orientation_relative_to_base_xyzw']
                        print(f"        obj_ori_rel_list_xyzw: {obj_ori_rel_list_xyzw}")
                        
                        # 检查obj_ori_rel_list_xyzw是否是包含4个元素的列表/元组
                        if not (isinstance(obj_ori_rel_list_xyzw, (list, tuple)) and len(obj_ori_rel_list_xyzw) == 4):
                            raise ValueError(f"object_orientation_relative_to_base_xyzw 格式不正确: {obj_ori_rel_list_xyzw}")

                        obj_ori_rel_quat_obj = gymapi.Quat(*obj_ori_rel_list_xyzw)
                        obj_ori_rel_rpy_list = convert_gymapi_quat_to_rpy_list(obj_ori_rel_quat_obj)
                        print(f"        obj_ori_rel_rpy_list: {obj_ori_rel_rpy_list}")

                        print(f"      轨迹 {i+1}, 帧 {frame_idx}: 正在拼接 state_vector...")
                        state_vector = np.concatenate([
                            joint_pos_np, joint_vel_np,
                            np.array(obj_pos_rel_list, dtype=np.float32),
                            np.array(obj_ori_rel_list_xyzw, dtype=np.float32),
                            np.array(obj_ori_rel_rpy_list, dtype=np.float32),
                            gripper_state_np
                        ]).astype(np.float32)
                        print(f"      轨迹 {i+1}, 帧 {frame_idx}: state_vector 定义成功, shape: {state_vector.shape}")
                        
                        # --- 开始定义 action_vector 及其组件 ---
                        print(f"      轨迹 {i+1}, 帧 {frame_idx}: 正在定义 action_vector 组件...")
                        if 'eef_target_position_relative_to_base' not in frame_data: raise KeyError("'eef_target_position_relative_to_base' not in frame_data")
                        target_eef_p_rel_list = frame_data['eef_target_position_relative_to_base']
                        
                        if 'eef_target_orientation_relative_to_base_xyzw' not in frame_data: raise KeyError("'eef_target_orientation_relative_to_base_xyzw' not in frame_data")
                        target_eef_r_rel_list_xyzw = frame_data['eef_target_orientation_relative_to_base_xyzw']
                        
                        # gripper_openness_ratio 已在上面用于 state_vector, 这里直接使用 frame_data 中的值
                        if 'gripper_openness_ratio' not in frame_data: raise KeyError("'gripper_openness_ratio' for action not in frame_data")
                        target_gripper_ratio_val = frame_data['gripper_openness_ratio']

                        if not (isinstance(target_eef_r_rel_list_xyzw, (list, tuple)) and len(target_eef_r_rel_list_xyzw) == 4):
                             raise ValueError(f"eef_target_orientation_relative_to_base_xyzw 格式不正确: {target_eef_r_rel_list_xyzw}")

                        target_eef_r_relative_quat_obj = gymapi.Quat(*target_eef_r_rel_list_xyzw)
                        target_eef_r_relative_rpy_list = convert_gymapi_quat_to_rpy_list(target_eef_r_relative_quat_obj)
                        print(f"        target_eef_r_relative_rpy_list: {target_eef_r_relative_rpy_list}")

                        print(f"      轨迹 {i+1}, 帧 {frame_idx}: 正在拼接 action_vector...")
                        action_vector = np.concatenate([
                            np.array(target_eef_p_rel_list, dtype=np.float32),
                            np.array(target_eef_r_relative_rpy_list, dtype=np.float32),
                            np.array([target_gripper_ratio_val], dtype=np.float32)
                        ]).astype(np.float32)
                        print(f"      轨迹 {i+1}, 帧 {frame_idx}: action_vector 定义成功, shape: {action_vector.shape}")

                        # 只有当 state_vector 和 action_vector 都成功定义后，才添加所有数据
                        current_traj_pc_list.append(sampled_points_np)
                        current_traj_states_list.append(state_vector) # 这是您出错的行 (line 364)
                        current_traj_actions_list.append(action_vector)
                        # valid_frames_collected_in_traj 将在帧循环后通过 len(current_traj_pc_list) 确定

                    except KeyError as ke:
                        print(f"    !!!!!! 轨迹 {i+1}, 帧 {frame_idx}: 定义state/action组件时发生 KeyError: {ke}。此帧数据将不完整，不予添加。 !!!!!!")
                        # sampled_points_np 已经通过检查是非零的，但由于state/action无法构建，我们不能添加这一帧
                        # 需要从 current_traj_pc_list 中移除刚刚可能错误添加的 sampled_points_np (如果添加逻辑在try之外)
                        # 或者确保只有在所有数据都有效时才添加。
                        # 为了简单，如果这里出错，我们确保这帧的任何部分都不被加入最终列表。
                        # 由于 current_traj_pc_list.append(sampled_points_np) 在 try 块之外，
                        # 如果这里发生 KeyError，点云已被添加，但状态和动作未被添加，这会导致数据不一致。
                        # 因此，最好将 current_traj_pc_list.append 也移入这个 try 块，
                        # 或者在捕获到异常时，如果之前添加了点云，则将其移除。

                        # 更安全的做法是将所有append操作都放在try成功之后：
                        # (上面的代码已调整，将append移到try块内，在所有vector定义成功后)
                        pass # 已经打印错误，此帧不会被完整记录

                    except ValueError as ve:
                        print(f"    !!!!!! 轨迹 {i+1}, 帧 {frame_idx}: 定义state/action组件时发生 ValueError (通常与Quat构造有关): {ve}。此帧数据将不完整，不予添加。 !!!!!!")
                        pass

                    except Exception as e_state_action:
                        print(f"    !!!!!! 轨迹 {i+1}, 帧 {frame_idx}: 定义state/action组件时发生未知错误: {e_state_action}。此帧数据将不完整，不予添加。 !!!!!!")
                        traceback.print_exc()
                        pass

            
            joint_pos_np=np.array(frame_data['joint_positions'],dtype=np.float32)
            joint_vel_np=np.array(frame_data['joint_velocities'],dtype=np.float32)
            gripper_state_np=np.array([frame_data['gripper_openness_ratio']],dtype=np.float32)
            obj_pos_rel_list=frame_data['object_position_relative_to_base']
            obj_ori_rel_list_xyzw=frame_data['object_orientation_relative_to_base_xyzw']
            obj_ori_rel_quat_obj=gymapi.Quat(*obj_ori_rel_list_xyzw)
            obj_ori_rel_rpy_list=convert_gymapi_quat_to_rpy_list(obj_ori_rel_quat_obj)
            state_vector=np.concatenate([
                joint_pos_np, joint_vel_np,
                np.array(obj_pos_rel_list,dtype=np.float32),
                np.array(obj_ori_rel_list_xyzw,dtype=np.float32),
                np.array(obj_ori_rel_rpy_list,dtype=np.float32),
                gripper_state_np
            ]).astype(np.float32)
            current_traj_states_list.append(state_vector)

            target_eef_p_rel_list=frame_data['eef_target_position_relative_to_base']
            target_eef_r_rel_list_xyzw=frame_data['eef_target_orientation_relative_to_base_xyzw']
            target_gripper_ratio_val=frame_data['gripper_openness_ratio']
            target_eef_r_relative_quat_obj=gymapi.Quat(*target_eef_r_rel_list_xyzw)
            target_eef_r_relative_rpy_list=convert_gymapi_quat_to_rpy_list(target_eef_r_relative_quat_obj)
            action_vector=np.concatenate([
                np.array(target_eef_p_rel_list,dtype=np.float32),
                np.array(target_eef_r_relative_rpy_list,dtype=np.float32),
                np.array([target_gripper_ratio_val],dtype=np.float32)
            ]).astype(np.float32)
            current_traj_actions_list.append(action_vector)
            # valid_frames_collected_in_traj 将在帧循环后通过 len(current_traj_pc_list) 确定
        # --- 单条轨迹内帧循环结束 ---

        valid_frames_collected_in_traj = len(current_traj_pc_list) # 实际收集到的带有点云的帧数

        # --- 新增：轨迹级检查 (基于全零点云帧的比例) ---
        if pcd_processing_attempted_in_traj > 0: # 避免除以零
            zero_pcd_frame_ratio = zero_pcd_frames_in_traj / pcd_processing_attempted_in_traj
            if zero_pcd_frame_ratio > 0.30:
                print(f"    警告: 轨迹 {i+1} ({os.path.basename(traj_file_path)}) 因点云全零帧过多而被放弃。")
                print(f"      (尝试处理 {pcd_processing_attempted_in_traj} 帧, 其中 {zero_pcd_frames_in_traj} 帧点云为零, "
                      f"比例: {zero_pcd_frame_ratio*100:.2f}%)")
                skipped_traj_count += 1
                continue # 跳到下一条轨迹
        # --- 轨迹级检查结束 ---
        
        # 日志：关于初步跳过的帧和最终有效帧
        if initial_frames_skipped_in_traj > 0 or zero_pcd_frames_in_traj > 0 :
             print(f"    轨迹 {i+1} ({os.path.basename(traj_file_path)}): "
                   f"初步跳过 {initial_frames_skipped_in_traj} 帧, "
                   f"点云处理尝试 {pcd_processing_attempted_in_traj} 帧, "
                   f"其中点云为零 {zero_pcd_frames_in_traj} 帧. "
                   f"最终有效(非零点云)帧数: {valid_frames_collected_in_traj}.")
        
        if valid_frames_collected_in_traj < MIN_FRAMES_PER_TRAJECTORY:
            if valid_frames_collected_in_traj > 0 or initial_frames_skipped_in_traj > 0 or zero_pcd_frames_in_traj > 0 :
                print(f"    轨迹 {i+1} ({os.path.basename(traj_file_path)}): 有效(非零点云)帧数 {valid_frames_collected_in_traj} < {MIN_FRAMES_PER_TRAJECTORY}，跳过此轨迹。")
            skipped_traj_count += 1
            continue
        
        if valid_frames_collected_in_traj > 0: # 确保确实有数据可写
            traj_pc_np = np.array(current_traj_pc_list, dtype=np.float32)
            traj_state_np = np.array(current_traj_states_list, dtype=np.float32)
            traj_action_np = np.array(current_traj_actions_list, dtype=np.float32)

            if not first_trajectory_successfully_written:
                pc_shape = (0, NUM_POINTS_EXPECTED, 6); pc_chunks = (1, NUM_POINTS_EXPECTED, 6)
                pc_dataset = data_group.require_dataset('point_cloud', shape=pc_shape, chunks=pc_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                
                state_dim = traj_state_np.shape[1]; state_shape = (0, state_dim); state_chunks = (min(100, valid_frames_collected_in_traj), state_dim)
                state_dataset = data_group.require_dataset('state', shape=state_shape, chunks=state_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                
                action_dim = traj_action_np.shape[1]; action_shape = (0, action_dim); action_chunks = (min(100, valid_frames_collected_in_traj), action_dim)
                action_dataset = data_group.require_dataset('action', shape=action_shape, chunks=action_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                
                print(f"    Zarr数据集已初始化。State dim: {state_dim}, Action dim: {action_dim}")
                first_trajectory_successfully_written = True
            
            try:
                pc_dataset.append(traj_pc_np)
                state_dataset.append(traj_state_np)
                action_dataset.append(traj_action_np)
            except Exception as e_zarr_append_main:
                print(f"    错误: 追加数据到Zarr失败 (轨迹 {i+1}): {e_zarr_append_main}")
                skipped_traj_count += 1 # 将此轨迹也视为跳过
                # 从 episode_ends_list 中移除可能已添加的条目 (如果适用，但此处在append后添加)
                continue 

            current_total_frames_in_zarr += valid_frames_collected_in_traj
            episode_ends_list.append(current_total_frames_in_zarr - 1)
            processed_traj_count += 1

            if processed_traj_count % 10 == 0 or i == len(trajectory_files) -1 :
                 print(f"  已处理 {processed_traj_count} 条有效轨迹。当前Zarr中总有效帧数: {current_total_frames_in_zarr}")
    
    if not first_trajectory_successfully_written:
        print("[主循环] 错误：没有有效轨迹被处理并写入Zarr文件。")
        if os.path.exists(OUTPUT_ZARR_PATH) and os.path.isdir(OUTPUT_ZARR_PATH) and not os.listdir(OUTPUT_ZARR_PATH): 
            try: shutil.rmtree(OUTPUT_ZARR_PATH); print(f"  已删除空的Zarr目录: {OUTPUT_ZARR_PATH}")
            except Exception as e_del_empty_zarr: print(f"  删除空Zarr目录失败: {e_del_empty_zarr}")
        return

    if 'episode_ends' in meta_group: del meta_group['episode_ends'] 
    if episode_ends_list: 
        meta_group.create_dataset('episode_ends', data=np.array(episode_ends_list, dtype=np.int64), chunks=(min(1000, len(episode_ends_list)),))

    print(f"\n[计时] Zarr文件处理主循环耗时: {time.time() - main_loop_start_time:.2f} 秒")
    print("\n数据转换完成。")
    print(f"  总轨迹文件数: {len(trajectory_files)}")
    print(f"  成功处理并写入Zarr的轨迹数: {processed_traj_count}")
    print(f"  跳过的轨迹数 (因各种原因): {skipped_traj_count}")
    print(f"  Zarr中总帧数: {current_total_frames_in_zarr}")

    if pc_dataset is not None : print(f"点云数据集形状: {pc_dataset.shape}, 分块: {pc_dataset.chunks}")
    if state_dataset is not None : print(f"状态数据集形状: {state_dataset.shape}, 分块: {state_dataset.chunks}")
    if action_dataset is not None : print(f"动作数据集形状: {action_dataset.shape}, 分块: {action_dataset.chunks}")
    if episode_ends_list: print(f"片段结束点 ({len(episode_ends_list)}条轨迹): {meta_group['episode_ends'][-5:] if 'episode_ends' in meta_group and len(episode_ends_list) > 5 else (meta_group['episode_ends'][:] if 'episode_ends' in meta_group else 'N/A')}")

    print(f"\n[计时] 脚本总运行耗时: {time.time() - overall_start_time:.2f} 秒")
    print(f"\n数据写入Zarr文件: {OUTPUT_ZARR_PATH}");
    print("Zarr文件结构 (顶层):");
    try:
        for item_name, item_obj in zarr_root.groups():
            print(f"  - Group: {item_name}, Path: {item_obj.path}")
            for arr_name, arr_obj in item_obj.arrays():
                 print(f"    - Array: {arr_name}, Shape: {arr_obj.shape}, Chunks: {arr_obj.chunks}, Dtype: {arr_obj.dtype}")
    except Exception as e_zarr_tree:
        print(f"打印Zarr树信息时出错: {e_zarr_tree}")

if __name__ == '__main__':
    main()
