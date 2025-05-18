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
import numpy as np
import cv2 # 主脚本中预处理图像也需要
from ultralytics import FastSAM # 主脚本中加载模型需要
import torch # 主脚本中设置DEVICE和预热需要
import time
import traceback
import open3d as o3d
import zarr
from typing import Tuple, Dict
from scipy.spatial.transform import Rotation as R_scipy
import shutil
# 从模型模块导入核心函数
# 假设这些函数存在于 model.point_cloud_crop 模块中
# from model.point_cloud_crop import create_point_cloud_from_depth_internal
# from model.point_cloud_crop import save_point_cloud_to_html
from model.point_cloud_crop import segment_and_crop_with_fastsam as external_point_cloud_processor
import sys

# --- DEVICE 定义修正 ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
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
OUTPUT_ZARR_PATH = '/data1/zhoufang/dataset_output/processed_data.zarr'
DATA_DIR = '/data1/zhoufang/dataset/collected_data'
MIN_FRAMES_PER_TRAJECTORY = 50
IMAGE_WIDTH_CONFIG = 640
IMAGE_HEIGHT_CONFIG = 480
FASTSAM_MODEL_PATH = './model/FastSAM-x.pt'
DEFAULT_TEXT_PROMPT_HAND = "robotic gripper with fingers"
DEFAULT_TEXT_PROMPT_OBJECT = "table"
DEPTH_SCALE = 1.0
print(f"重要提示: DEPTH_SCALE 设置为 {DEPTH_SCALE}。这假设您的原始深度数据单位是米。")
debug_first_frame_only = False # 通常在完整处理时设为 False

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
    euler_zyx = quat_obj.to_euler_zyx(); return [euler_zyx[2], euler_zyx[1], euler_zyx[0]] # RPY is X, Y, Z order of rotations

def main():
    overall_start_time = time.time()
    print(f"将使用图像尺寸: Width={IMAGE_WIDTH_CONFIG}, Height={IMAGE_HEIGHT_CONFIG}")
    print(f"处理点云将使用设备: {DEVICE}")

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
            # 使用符合模型预期的输入尺寸
            dummy_img_warmup = np.zeros((IMAGE_HEIGHT_CONFIG, IMAGE_WIDTH_CONFIG, 3), dtype=np.uint8)
            # FastSAM的predict方法可能需要source参数，确保传递
            _ = fastsam_model_instance(dummy_img_warmup, device=DEVICE, retina_masks=True, imgsz=IMAGE_WIDTH_CONFIG, verbose=False, conf=0.1, texts=["a generic object"])
            torch.cuda.synchronize() # 确保预热操作完成
            print(f"  FastSAM模型预热完成。耗时: {time.time() - t_warmup_start:.2f} 秒")
        except Exception as e_warmup:
            print(f"  FastSAM模型预热时出错: {e_warmup}")
            print(f"  请检查选定GPU (通过 CUDA_VISIBLE_DEVICES 设置为 {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}) 是否有足够显存。")
            traceback.print_exc()

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
        print("[调试模式] 'debug_first_frame_only' 为 True，脚本将只处理第一条轨迹的第一帧并退出。")
        # 此处应有调试特定帧的逻辑，为简洁起见，假设它与主循环逻辑相似但只运行一次
        # 这里可以复制主循环中处理单帧的部分逻辑用于测试
        # ... (调试代码) ...
        print("调试模式结束。")
        return

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
    first_trajectory_successfully_written = False # 用于标记是否已初始化Zarr数据集

    main_loop_conf_hand = 0.25
    main_loop_conf_object = 0.25
    print(f"[主循环] FastSAM置信度: 手={main_loop_conf_hand}, 物体={main_loop_conf_object}")
    main_loop_imgsz_w = IMAGE_WIDTH_CONFIG
    main_loop_imgsz_h = IMAGE_HEIGHT_CONFIG

    processed_traj_count = 0
    skipped_traj_count = 0

    for i, traj_file_path in enumerate(trajectory_files):
        current_traj_dir = os.path.dirname(traj_file_path)
        print(f"\n开始处理轨迹 {i+1}/{len(trajectory_files)}: {os.path.basename(traj_file_path)}")
        try:
            with open(traj_file_path, 'r') as f_json: raw_trajectory_data = json.load(f_json)
            if not isinstance(raw_trajectory_data, list) or not raw_trajectory_data:
                print(f"    警告: 轨迹 {i+1} JSON 内容无效或为空，跳过。")
                skipped_traj_count += 1
                continue
        except Exception as e_load_json_main:
            print(f"    警告: 加载轨迹 {i+1} JSON ({os.path.basename(traj_file_path)}) 失败: {e_load_json_main}，跳过。")
            skipped_traj_count += 1
            continue

        current_traj_pc_list = []
        current_traj_states_list = []
        current_traj_actions_list = []

        initial_frames_skipped_in_traj = 0
        pcd_processing_attempted_in_traj = 0
        zero_pcd_frames_in_traj = 0

        for frame_idx, frame_data in enumerate(raw_trajectory_data):
            # print(f"  处理轨迹 {i+1}, 帧 {frame_idx+1}/{len(raw_trajectory_data)}")
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
            missing_keys_info = []
            for key, exp_type in required_keys.items():
                val = frame_data.get(key)
                if val is None:
                    valid_frame = False; missing_keys_info.append(f"缺失键'{key}'"); break
                if isinstance(exp_type, tuple):
                    if not any(isinstance(val, t) for t in exp_type):
                        valid_frame = False; missing_keys_info.append(f"键'{key}'类型错误(应为{exp_type},实为{type(val)})"); break
                elif not isinstance(val, exp_type):
                    valid_frame = False; missing_keys_info.append(f"键'{key}'类型错误(应为{exp_type},实为{type(val)})"); break
            if not valid_frame:
                # print(f"    轨迹 {i+1}, 帧 {frame_idx}: 数据格式无效 ({'; '.join(missing_keys_info)})，跳过此帧。")
                initial_frames_skipped_in_traj += 1
                continue

            k_matrix = np.array(frame_data['camera_K_matrix'], dtype=np.float32)
            if k_matrix.shape != (3,3):
                # print(f"    轨迹 {i+1}, 帧 {frame_idx}: K矩阵形状无效，跳过此帧。")
                initial_frames_skipped_in_traj += 1; continue

            depth_path = os.path.join(current_traj_dir, frame_data['depth_image_path'])
            rgb_path = os.path.join(current_traj_dir, frame_data['rgb_image_path'])
            try:
                depth_np = np.load(depth_path); rgb_np_loaded = np.load(rgb_path)
                if rgb_np_loaded.shape[:2] != (IMAGE_HEIGHT_CONFIG, IMAGE_WIDTH_CONFIG) or \
                   depth_np.shape[:2] != (IMAGE_HEIGHT_CONFIG, IMAGE_WIDTH_CONFIG):
                    # print(f"    轨迹 {i+1}, 帧 {frame_idx}: 图像/深度图尺寸不匹配，跳过此帧。")
                    initial_frames_skipped_in_traj += 1; continue
                if not np.issubdtype(depth_np.dtype, np.floating): depth_np = depth_np.astype(np.float32)
            except FileNotFoundError:
                # print(f"    轨迹 {i+1}, 帧 {frame_idx}: 深度图或RGB图像文件未找到，跳过此帧。")
                initial_frames_skipped_in_traj += 1; continue
            except Exception as e_load_img:
                # print(f"    轨迹 {i+1}, 帧 {frame_idx}: 加载图像时出错 ({e_load_img})，跳过此帧。")
                initial_frames_skipped_in_traj += 1; continue

            rgb_input_main_uint8 = rgb_np_loaded.copy()
            if not np.issubdtype(rgb_input_main_uint8.dtype, np.uint8):
                if np.issubdtype(rgb_input_main_uint8.dtype, np.floating) and rgb_input_main_uint8.max()<=1.0 and rgb_input_main_uint8.min()>=0.0:
                    rgb_input_main_uint8=(rgb_input_main_uint8*255).astype(np.uint8)
                elif np.issubdtype(rgb_input_main_uint8.dtype, np.integer) and rgb_input_main_uint8.max() > 255: # e.g. uint16
                     rgb_input_main_uint8 = (rgb_input_main_uint8 / np.max(rgb_input_main_uint8) * 255).astype(np.uint8) # Normalize and convert
                else:
                    rgb_input_main_uint8=np.clip(rgb_input_main_uint8,0,255).astype(np.uint8)


            if rgb_input_main_uint8.ndim == 3 and rgb_input_main_uint8.shape[2] == 4:
                rgb_bgr_main = cv2.cvtColor(rgb_input_main_uint8, cv2.COLOR_RGBA2BGR)
            elif rgb_input_main_uint8.ndim == 3 and rgb_input_main_uint8.shape[2] == 3:
                rgb_bgr_main = cv2.cvtColor(rgb_input_main_uint8, cv2.COLOR_RGB2BGR)
            elif rgb_input_main_uint8.ndim == 2:
                rgb_bgr_main = cv2.cvtColor(rgb_input_main_uint8, cv2.COLOR_GRAY2BGR)
            else:
                # print(f"    轨迹 {i+1}, 帧 {frame_idx}: RGB图像格式无法转换为BGR，跳过此帧。")
                initial_frames_skipped_in_traj += 1; continue

            pcd_processing_attempted_in_traj += 1
            sampled_points_np = np.zeros((NUM_POINTS_EXPECTED, 6), dtype=np.float32) # 默认为全零

            try:
                # print(f"    轨迹 {i+1}, 帧 {frame_idx}: 调用 external_point_cloud_processor...")
                processed_o3d_pcd = external_point_cloud_processor(
                    rgb_image_input=rgb_bgr_main, # FastSAM通常期望BGR格式
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
                    timing_prefix=f"Traj{i+1}_Frm{frame_idx}_"
                )

                if processed_o3d_pcd and processed_o3d_pcd.has_points():
                    points_xyz = np.asarray(processed_o3d_pcd.points, dtype=np.float32)
                    if processed_o3d_pcd.has_colors():
                        points_rgb = np.asarray(processed_o3d_pcd.colors, dtype=np.float32)
                    else:
                        points_rgb = np.full((points_xyz.shape[0], 3), 0.5, dtype=np.float32) # 默认灰色

                    if points_xyz.shape[0] > 0:
                        points_xyzrgb = np.hstack((points_xyz, points_rgb))
                        num_curr = points_xyzrgb.shape[0]

                        if num_curr >= NUM_POINTS_EXPECTED: # 等于或大于期望点数，进行下采样
                            indices = np.random.choice(num_curr, NUM_POINTS_EXPECTED, replace=False)
                            sampled_points_np = points_xyzrgb[indices]
                        else: # num_curr < NUM_POINTS_EXPECTED and num_curr > 0
                            padding_indices = np.random.choice(num_curr, NUM_POINTS_EXPECTED - num_curr, replace=True)
                            sampled_points_np = np.vstack((points_xyzrgb, points_xyzrgb[padding_indices]))
                # else:
                    # print(f"    轨迹 {i+1}, 帧 {frame_idx}: external_point_cloud_processor 返回 None 或空点云。sampled_points_np 保持全零。")

            except Exception as e_pcd_proc_main_loop:
                print(f"    !!!!!! 轨迹 {i+1}, 帧 {frame_idx}: 在点云处理或采样中发生异常: {e_pcd_proc_main_loop} !!!!!!")
                traceback.print_exc()
                # sampled_points_np 保持为全零

            # 点云检查 (移到点云处理逻辑之后)
            is_effectively_zero = np.all(np.abs(sampled_points_np) < 1e-9)

            if is_effectively_zero:
                # print(f"    警告: 轨迹 {i+1}, 帧 {frame_idx}: 点云数据被判定为全零或几乎全零，跳过此帧。")
                zero_pcd_frames_in_traj += 1
                continue
            else:
                # 点云有效，尝试构建 state 和 action
                try:
                    # --- 定义 state_vector ---
                    joint_pos_np = np.array(frame_data['joint_positions'], dtype=np.float32)
                    joint_vel_np = np.array(frame_data['joint_velocities'], dtype=np.float32)
                    gripper_state_np = np.array([frame_data['gripper_openness_ratio']], dtype=np.float32)
                    obj_pos_rel_list = frame_data['object_position_relative_to_base']
                    obj_ori_rel_list_xyzw = frame_data['object_orientation_relative_to_base_xyzw']

                    if not (isinstance(obj_ori_rel_list_xyzw, (list, tuple)) and len(obj_ori_rel_list_xyzw) == 4):
                        raise ValueError(f"object_orientation_relative_to_base_xyzw 格式不正确: {obj_ori_rel_list_xyzw}")
                    obj_ori_rel_quat_obj = gymapi.Quat(*obj_ori_rel_list_xyzw)
                    obj_ori_rel_rpy_list = convert_gymapi_quat_to_rpy_list(obj_ori_rel_quat_obj)

                    state_vector = np.concatenate([
                        joint_pos_np, joint_vel_np,
                        np.array(obj_pos_rel_list, dtype=np.float32),
                        np.array(obj_ori_rel_list_xyzw, dtype=np.float32), # 保存四元数
                        np.array(obj_ori_rel_rpy_list, dtype=np.float32),   # 也保存RPY，根据需要选择使用
                        gripper_state_np
                    ]).astype(np.float32)

                    # --- 定义 action_vector ---
                    target_eef_p_rel_list = frame_data['eef_target_position_relative_to_base']
                    target_eef_r_rel_list_xyzw = frame_data['eef_target_orientation_relative_to_base_xyzw']
                    target_gripper_ratio_val = frame_data['gripper_openness_ratio']

                    if not (isinstance(target_eef_r_rel_list_xyzw, (list, tuple)) and len(target_eef_r_rel_list_xyzw) == 4):
                         raise ValueError(f"eef_target_orientation_relative_to_base_xyzw 格式不正确: {target_eef_r_rel_list_xyzw}")
                    target_eef_r_relative_quat_obj = gymapi.Quat(*target_eef_r_rel_list_xyzw)
                    target_eef_r_relative_rpy_list = convert_gymapi_quat_to_rpy_list(target_eef_r_relative_quat_obj)

                    action_vector = np.concatenate([
                        np.array(target_eef_p_rel_list, dtype=np.float32),
                        np.array(target_eef_r_relative_rpy_list, dtype=np.float32), # 使用RPY作为动作的一部分
                        # 如果需要四元数作为动作，则使用 np.array(target_eef_r_rel_list_xyzw, dtype=np.float32)
                        np.array([target_gripper_ratio_val], dtype=np.float32)
                    ]).astype(np.float32)

                    # 只有当点云、state_vector 和 action_vector 都成功定义后，才添加所有数据
                    current_traj_pc_list.append(sampled_points_np)
                    current_traj_states_list.append(state_vector)
                    current_traj_actions_list.append(action_vector)

                except KeyError as ke:
                    print(f"    !!!!!! 轨迹 {i+1}, 帧 {frame_idx}: 定义state/action组件时发生 KeyError: {ke}。此帧数据将不完整，不予添加。 !!!!!!")
                    pass
                except ValueError as ve:
                    print(f"    !!!!!! 轨迹 {i+1}, 帧 {frame_idx}: 定义state/action组件时发生 ValueError: {ve}。此帧数据将不完整，不予添加。 !!!!!!")
                    pass
                except Exception as e_state_action:
                    print(f"    !!!!!! 轨迹 {i+1}, 帧 {frame_idx}: 定义state/action组件时发生未知错误: {e_state_action}。此帧数据将不完整，不予添加。 !!!!!!")
                    traceback.print_exc()
                    pass

        # --- 单条轨迹内帧循环结束 ---

        valid_frames_collected_in_traj = len(current_traj_pc_list)
        # 确保三个列表长度一致
        if not (len(current_traj_pc_list) == len(current_traj_states_list) == len(current_traj_actions_list)):
            print(f"    !!!!!! 严重错误: 轨迹 {i+1} 内部数据列表长度不一致! "
                  f"PCD: {len(current_traj_pc_list)}, State: {len(current_traj_states_list)}, Action: {len(current_traj_actions_list)}")
            print(f"    跳过此轨迹以避免数据损坏。")
            skipped_traj_count += 1
            continue


        if pcd_processing_attempted_in_traj > 0:
            zero_pcd_frame_ratio = zero_pcd_frames_in_traj / pcd_processing_attempted_in_traj
            if zero_pcd_frame_ratio > 0.30: # 如果超过30%的点云处理尝试结果是全零，则放弃该轨迹
                print(f"    警告: 轨迹 {i+1} ({os.path.basename(traj_file_path)}) 因点云全零帧过多而被放弃。")
                print(f"      (尝试处理 {pcd_processing_attempted_in_traj} 帧, 其中 {zero_pcd_frames_in_traj} 帧点云为零, "
                      f"比例: {zero_pcd_frame_ratio*100:.2f}%)")
                skipped_traj_count += 1
                continue

        if initial_frames_skipped_in_traj > 0 or zero_pcd_frames_in_traj > 0 :
             print(f"    轨迹 {i+1} ({os.path.basename(traj_file_path)}) 统计: "
                   f"初步跳过 {initial_frames_skipped_in_traj} 帧 (格式/文件问题), "
                   f"点云处理尝试 {pcd_processing_attempted_in_traj} 帧, "
                   f"其中点云为零 {zero_pcd_frames_in_traj} 帧. "
                   f"最终有效帧数 (点云、状态、动作均有效): {valid_frames_collected_in_traj}.")

        if valid_frames_collected_in_traj < MIN_FRAMES_PER_TRAJECTORY:
            if valid_frames_collected_in_traj > 0 or initial_frames_skipped_in_traj > 0 or zero_pcd_frames_in_traj > 0 : # 仅当有活动时打印
                print(f"    轨迹 {i+1} ({os.path.basename(traj_file_path)}): 有效帧数 {valid_frames_collected_in_traj} < 最低要求 {MIN_FRAMES_PER_TRAJECTORY}，跳过此轨迹。")
            else: # 如果轨迹完全为空或所有帧一开始就被跳过
                print(f"    轨迹 {i+1} ({os.path.basename(traj_file_path)}): 没有收集到有效帧数据，跳过此轨迹。")
            skipped_traj_count += 1
            continue

        # 只有当确实有有效数据可写时才继续
        if valid_frames_collected_in_traj > 0:
            traj_pc_np = np.array(current_traj_pc_list, dtype=np.float32)
            traj_state_np = np.array(current_traj_states_list, dtype=np.float32)
            traj_action_np = np.array(current_traj_actions_list, dtype=np.float32)

            if not first_trajectory_successfully_written:
                # 初始化Zarr数据集，只在第一次写入有效轨迹时执行
                pc_shape = (0, NUM_POINTS_EXPECTED, 6); pc_chunks = (1, NUM_POINTS_EXPECTED, 6) # 针对点云优化chunks
                pc_dataset = data_group.require_dataset('point_cloud', shape=pc_shape, chunks=pc_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))

                state_dim = traj_state_np.shape[1]; state_shape = (0, state_dim); state_chunks = (min(100, valid_frames_collected_in_traj), state_dim) # 较小的chunks适合state/action
                state_dataset = data_group.require_dataset('state', shape=state_shape, chunks=state_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))

                action_dim = traj_action_np.shape[1]; action_shape = (0, action_dim); action_chunks = (min(100, valid_frames_collected_in_traj), action_dim)
                action_dataset = data_group.require_dataset('action', shape=action_shape, chunks=action_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))

                print(f"    Zarr数据集已初始化。State dim: {state_dim}, Action dim: {action_dim}")
                first_trajectory_successfully_written = True

            # 确保数据集已创建 (如果不是第一次写入，则它们应该已存在)
            if pc_dataset is None: pc_dataset = data_group['point_cloud']
            if state_dataset is None: state_dataset = data_group['state']
            if action_dataset is None: action_dataset = data_group['action']

            try:
                pc_dataset.append(traj_pc_np)
                state_dataset.append(traj_state_np)
                action_dataset.append(traj_action_np)
            except Exception as e_zarr_append_main:
                print(f"    错误: 追加数据到Zarr失败 (轨迹 {i+1}): {e_zarr_append_main}")
                skipped_traj_count += 1
                # 发生错误时，不更新 episode_ends_list 和 current_total_frames_in_zarr
                continue # 跳到下一条轨迹

            current_total_frames_in_zarr += valid_frames_collected_in_traj
            # #############################################
            # ############### 修改点 #####################
            # #############################################
            episode_ends_list.append(current_total_frames_in_zarr) # 存储的是累积总帧数
            # #############################################
            processed_traj_count += 1
            print(f"    轨迹 {i+1} 处理完毕并添加。有效帧数: {valid_frames_collected_in_traj}。当前Zarr总有效帧数: {current_total_frames_in_zarr}")

            if processed_traj_count > 0 and (processed_traj_count % 10 == 0 or i == len(trajectory_files) -1) :
                 print(f"  已处理 {processed_traj_count} 条有效轨迹。")

    if not first_trajectory_successfully_written: # 如果没有任何轨迹被成功写入
        print("[主循环] 错误：没有有效轨迹被处理并写入Zarr文件。")
        # 如果Zarr目录已创建但为空，可以考虑删除它
        if os.path.exists(OUTPUT_ZARR_PATH) and os.path.isdir(OUTPUT_ZARR_PATH):
            try:
                # 检查目录是否为空 (更可靠的方式是尝试列出内容)
                if not any(os.scandir(OUTPUT_ZARR_PATH)):
                    shutil.rmtree(OUTPUT_ZARR_PATH)
                    print(f"  已删除空的Zarr目录: {OUTPUT_ZARR_PATH}")
            except Exception as e_del_empty_zarr:
                print(f"  删除空Zarr目录时出错: {e_del_empty_zarr}")
        return

    # 保存 episode_ends 元数据
    if 'episode_ends' in meta_group: del meta_group['episode_ends'] # 如果已存在则先删除
    if episode_ends_list: # 确保列表不为空
        meta_group.create_dataset('episode_ends', data=np.array(episode_ends_list, dtype=np.int64), chunks=(min(1000, len(episode_ends_list)),)) # 适合 episode_ends 的 chunks
    else:
        print("警告: episode_ends_list 为空，没有元数据被保存到 meta/episode_ends。")


    print(f"\n[计时] Zarr文件处理主循环耗时: {time.time() - main_loop_start_time:.2f} 秒")
    print("\n数据转换完成。")
    print(f"  总轨迹文件数: {len(trajectory_files)}")
    print(f"  成功处理并写入Zarr的轨迹数: {processed_traj_count}")
    print(f"  跳过的轨迹数 (因各种原因): {skipped_traj_count}")
    print(f"  Zarr中总帧数: {current_total_frames_in_zarr}")

    if pc_dataset is not None : print(f"点云数据集最终形状: {pc_dataset.shape}, 分块: {pc_dataset.chunks}")
    if state_dataset is not None : print(f"状态数据集最终形状: {state_dataset.shape}, 分块: {state_dataset.chunks}")
    if action_dataset is not None : print(f"动作数据集最终形状: {action_dataset.shape}, 分块: {action_dataset.chunks}")
    
    if episode_ends_list and 'episode_ends' in meta_group and meta_group['episode_ends'].size > 0:
        print(f"片段结束点 ({len(episode_ends_list)}条轨迹) 最后5条: {meta_group['episode_ends'][-5:]}")
    elif episode_ends_list:
        print(f"片段结束点 ({len(episode_ends_list)}条轨迹) 最后5条: {np.array(episode_ends_list, dtype=np.int64)[-5:]} (元数据中可能未正确写入或为空)")
    else:
        print("片段结束点列表为空。")


    print(f"\n[计时] 脚本总运行耗时: {time.time() - overall_start_time:.2f} 秒")
    if first_trajectory_successfully_written: # 仅当有数据写入时才打印成功
        print(f"\n数据成功写入Zarr文件: {OUTPUT_ZARR_PATH}");
        print("Zarr文件结构 (顶层):");
        try:
            # 打印最终的Zarr文件结构
            for item_name in zarr_root.keys():
                item_obj = zarr_root[item_name]
                print(f"  - {'Group' if isinstance(item_obj, zarr.hierarchy.Group) else 'Array'}: {item_name}")
                if isinstance(item_obj, zarr.hierarchy.Group):
                    for sub_item_name in item_obj.keys():
                        sub_item_obj = item_obj[sub_item_name]
                        print(f"    - {'Group' if isinstance(sub_item_obj, zarr.hierarchy.Group) else 'Array'}: {sub_item_name}, Shape: {getattr(sub_item_obj, 'shape', 'N/A')}, Chunks: {getattr(sub_item_obj, 'chunks', 'N/A')}, Dtype: {getattr(sub_item_obj, 'dtype', 'N/A')}")
        except Exception as e_zarr_tree:
            print(f"打印Zarr树信息时出错: {e_zarr_tree}")
    else:
        print(f"没有数据成功写入Zarr文件: {OUTPUT_ZARR_PATH}")


if __name__ == '__main__':
    main()