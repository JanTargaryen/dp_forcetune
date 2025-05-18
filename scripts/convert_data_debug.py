from isaacgym import gymapi # 假设此库可用
from myUtil import gpu_utl
import os

# --- 在所有其他 PyTorch 和 CUDA 相关导入之前设置环境变量 ---
# ... (这部分代码保持不变，为了简洁此处省略) ...
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
except ImportError:
    print("警告: 无法导入 'myUtil.gpu_utl'。将不会通过此工具选择GPU。")
    print("将尝试默认GPU或您之前设置的CUDA_VISIBLE_DEVICES（如果存在）。")
except Exception as e_gpu_select:
    print(f"警告: 使用 myUtil.gpu_utl 选择GPU时出错: {e_gpu_select}。将尝试默认行为。")
# -------------------------------------------------------------

import json
import numpy as np
import cv2
import torch
import time
import traceback
import open3d as o3d
import zarr
from typing import Tuple, Dict
from scipy.spatial.transform import Rotation as R_scipy
import shutil
import sys

try:
    from model.point_cloud_crop import create_point_cloud_from_depth_internal
    print("成功从 model.point_cloud_crop 导入 create_point_cloud_from_depth_internal")
except ImportError as e_import_custom:
    print(f"错误: 无法从 model.point_cloud_crop 导入函数: {e_import_custom}")
    print("请确保 model/point_cloud_crop.py 文件存在且路径正确。")
    sys.exit(1)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"PyTorch 检测到 CUDA 可用。")
    print(f"  将使用的PyTorch设备: {DEVICE}")
    try:
        print(f"  实际PyTorch设备名称: {torch.cuda.get_device_name(0)}")
    except Exception as e_dev_name:
        print(f"  获取选定GPU名称或显存时出错: {e_dev_name}")
else:
    DEVICE = torch.device("cpu")
    print("PyTorch 未检测到 CUDA 可用，将使用 CPU。")


NUM_POINTS_EXPECTED = 16384
OUTPUT_ZARR_PATH = '/data1/zhoufang/dataset_output/processed_data_single_traj.zarr' # <--- 修改输出路径名
DATA_DIR = '/data1/zhoufang/dataset/collected_data'
MIN_FRAMES_PER_TRAJECTORY = 10 # <--- 为单轨迹调试，可以适当降低此门槛
IMAGE_WIDTH_CONFIG = 640
IMAGE_HEIGHT_CONFIG = 480
DEPTH_SCALE_FACTOR_FOR_INTERNAL_FUNC = 1.0
print(f"重要提示: DEPTH_SCALE_FACTOR_FOR_INTERNAL_FUNC 设置为 {DEPTH_SCALE_FACTOR_FOR_INTERNAL_FUNC}。")
print(f"提示: 此版本将只处理找到的第一个轨迹文件。")
debug_first_frame_only = False # 这个可以保持False，因为我们用循环break来控制只处理一条轨迹

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
    print(f"找到 {len(trajectory_files)} 条轨迹文件。将只处理第一条。")


    print(f"\n[主循环] 开始处理轨迹...")
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

    processed_traj_count = 0
    skipped_traj_count = 0
    first_problematic_frame_debugged_main = False

    # #############################################################
    # ############### 修改点: 只处理第一个轨迹 #####################
    # #############################################################
    # for i, traj_file_path in enumerate(trajectory_files):
    if len(trajectory_files) > 0:
        i = 0 # 强制索引为0
        traj_file_path = trajectory_files[0] # 只取第一个文件
    # #############################################################
        current_traj_dir = os.path.dirname(traj_file_path)
        print(f"\n开始处理轨迹 {i+1}/{len(trajectory_files)} (仅处理此条): {os.path.basename(traj_file_path)}")
        try:
            with open(traj_file_path, 'r') as f_json: raw_trajectory_data = json.load(f_json)
            if not isinstance(raw_trajectory_data, list) or not raw_trajectory_data:
                print(f"    警告: 轨迹 {i+1} JSON 内容无效或为空，跳过。")
                skipped_traj_count += 1
                # continue # 在单轨迹模式下，如果第一个就失败，可以直接退出或处理
                raw_trajectory_data = [] # 置为空，避免后续循环出错
        except Exception as e_load_json_main:
            print(f"    警告: 加载轨迹 {i+1} JSON ({os.path.basename(traj_file_path)}) 失败: {e_load_json_main}，跳过。")
            skipped_traj_count += 1
            # continue
            raw_trajectory_data = []


        current_traj_pc_list = []
        current_traj_states_list = []
        current_traj_actions_list = []

        initial_frames_skipped_in_traj = 0
        pcd_processing_attempted_in_traj = 0
        zero_pcd_frames_in_traj = 0

        for frame_idx, frame_data in enumerate(raw_trajectory_data):
            # 详细调试打印逻辑 (should_print_debug_here 和 first_problematic_frame_debugged_main)
            # 将主要用于这个单轨迹的第一帧（如果它在之前就是有问题的）
            should_print_debug_here = (initial_frames_skipped_in_traj == 0 and not first_problematic_frame_debugged_main)

            required_keys = { # 省略定义，与之前一致
                'depth_image_path': str, 'rgb_image_path': str, 'camera_K_matrix': list,
                'joint_positions': list, 'joint_velocities': list,
                'eef_target_position_relative_to_base': list,
                'eef_target_orientation_relative_to_base_xyzw': list,
                'gripper_openness_ratio': (float, int),
                'object_position_relative_to_base': list,
                'object_orientation_relative_to_base_xyzw': list
            }
            valid_frame = True; missing_keys_info = []
            for key, exp_type in required_keys.items():
                val = frame_data.get(key)
                if val is None: valid_frame = False; missing_keys_info.append(f"缺失键'{key}'"); break
                if isinstance(exp_type, tuple):
                    if not any(isinstance(val, t) for t in exp_type): valid_frame = False; missing_keys_info.append(f"键'{key}'类型错误"); break
                elif not isinstance(val, exp_type): valid_frame = False; missing_keys_info.append(f"键'{key}'类型错误"); break
            if not valid_frame: initial_frames_skipped_in_traj += 1; continue

            k_matrix = np.array(frame_data['camera_K_matrix'], dtype=np.float32)
            if k_matrix.shape != (3,3): initial_frames_skipped_in_traj += 1; continue

            depth_path = os.path.join(current_traj_dir, frame_data['depth_image_path'])
            rgb_path = os.path.join(current_traj_dir, frame_data['rgb_image_path'])
            try:
                depth_np_original = np.load(depth_path); rgb_np_loaded = np.load(rgb_path)
                if rgb_np_loaded.shape[:2] != (IMAGE_HEIGHT_CONFIG, IMAGE_WIDTH_CONFIG) or \
                   depth_np_original.shape[:2] != (IMAGE_HEIGHT_CONFIG, IMAGE_WIDTH_CONFIG):
                    initial_frames_skipped_in_traj += 1; continue
                if not np.issubdtype(depth_np_original.dtype, np.floating):
                     depth_np_original = depth_np_original.astype(np.float32)
            except FileNotFoundError: initial_frames_skipped_in_traj += 1; continue
            except Exception as e_load_img:
                print(f"    轨迹 {i+1}, 帧 {frame_idx}: 加载图像时出错 ({e_load_img})，跳过此帧。")
                initial_frames_skipped_in_traj += 1; continue

            rgb_input_main_uint8 = rgb_np_loaded.copy() #处理颜色格式的代码与之前相同
            if not np.issubdtype(rgb_input_main_uint8.dtype, np.uint8):
                if np.issubdtype(rgb_input_main_uint8.dtype, np.floating) and rgb_input_main_uint8.max()<=1.0 and rgb_input_main_uint8.min()>=0.0:
                    rgb_input_main_uint8=(rgb_input_main_uint8*255).astype(np.uint8)
                elif np.issubdtype(rgb_input_main_uint8.dtype, np.integer) and rgb_input_main_uint8.max() > 255:
                     rgb_input_main_uint8 = (rgb_input_main_uint8 / np.max(rgb_input_main_uint8) * 255).astype(np.uint8) # Normalize and convert
                else: rgb_input_main_uint8=np.clip(rgb_input_main_uint8,0,255).astype(np.uint8)


            pcd_processing_attempted_in_traj += 1
            sampled_points_np = np.zeros((NUM_POINTS_EXPECTED, 6), dtype=np.float32)

            if should_print_debug_here:
                print(f"--- DEBUGGING FRAME: Trajectory {i+1} (file: {os.path.basename(traj_file_path)}), Frame Index in JSON {frame_idx} ---")
                print(f"Original depth_np_original shape: {depth_np_original.shape}, dtype: {depth_np_original.dtype}")
                # 检查深度值是否全为负，并尝试取绝对值进行处理
                if np.all(depth_np_original <= 0) and np.any(depth_np_original < 0): # 如果所有值都小于等于0，且至少有一个小于0
                    print(f"WARNING: All loaded depth values are non-positive. Min: {np.min(depth_np_original)}, Max: {np.max(depth_np_original)}")
                    print("Attempting to use absolute values for depth.")
                    depth_to_use_for_pcd = np.abs(depth_np_original)
                else:
                    depth_to_use_for_pcd = depth_np_original

                print(f"Depth to use for PCD (after potential abs): min: {np.nanmin(depth_to_use_for_pcd):.4f}, max: {np.nanmax(depth_to_use_for_pcd):.4f}, mean: {np.nanmean(depth_to_use_for_pcd):.4f}")
                print(f"Number of NaNs in depth_to_use_for_pcd: {np.isnan(depth_to_use_for_pcd).sum()}")
                print(f"Number of Infs in depth_to_use_for_pcd: {np.isinf(depth_to_use_for_pcd).sum()}")
                print(f"Number of zeros in depth_to_use_for_pcd: {(depth_to_use_for_pcd == 0).sum()}")
                print(f"rgb_input_main_uint8 shape: {rgb_input_main_uint8.shape}, dtype: {rgb_input_main_uint8.dtype}")
                print(f"k_matrix:\n{k_matrix}")
                print(f"DEPTH_SCALE_FACTOR_FOR_INTERNAL_FUNC: {DEPTH_SCALE_FACTOR_FOR_INTERNAL_FUNC}")
            else: # 非调试帧，正常处理深度
                if np.all(depth_np_original <= 0) and np.any(depth_np_original < 0):
                    depth_to_use_for_pcd = np.abs(depth_np_original)
                else:
                    depth_to_use_for_pcd = depth_np_original


            try:
                processed_o3d_pcd, _ = create_point_cloud_from_depth_internal(
                    depth_image=depth_to_use_for_pcd, # 使用处理过的深度
                    k_matrix=k_matrix,
                    depth_scale_factor=DEPTH_SCALE_FACTOR_FOR_INTERNAL_FUNC,
                    rgb_image=rgb_input_main_uint8,
                    image_height_config=IMAGE_HEIGHT_CONFIG,
                    image_width_config=IMAGE_WIDTH_CONFIG
                )

                if should_print_debug_here: # 调试打印（与之前版本相同）
                    if processed_o3d_pcd is None: print("DEBUG: processed_o3d_pcd is None after internal_create")
                    else:
                        print(f"DEBUG: processed_o3d_pcd.has_points(): {processed_o3d_pcd.has_points()}")
                        if processed_o3d_pcd.has_points():
                            raw_points_xyz = np.asarray(processed_o3d_pcd.points)
                            print(f"DEBUG: Number of points from internal_create: {len(raw_points_xyz)}")
                            if len(raw_points_xyz) > 0:
                                print(f"DEBUG: Raw points_xyz min: {np.min(raw_points_xyz, axis=0)}, max: {np.max(raw_points_xyz, axis=0)}")
                                if processed_o3d_pcd.has_colors():
                                     raw_colors = np.asarray(processed_o3d_pcd.colors)
                                     print(f"DEBUG: Raw colors (0-1 scale) min: {np.min(raw_colors, axis=0)}, max: {np.max(raw_colors, axis=0)}")
                            else: print("DEBUG: internal_create resulted in 0 points.")
                        else: print("DEBUG: internal_create created pcd but has no points.")


                if processed_o3d_pcd and processed_o3d_pcd.has_points(): #采样逻辑与之前相同
                    points_xyz = np.asarray(processed_o3d_pcd.points, dtype=np.float32)
                    if processed_o3d_pcd.has_colors():
                        points_rgb = np.asarray(processed_o3d_pcd.colors, dtype=np.float32) * 255.0
                        points_rgb = np.clip(points_rgb, 0, 255).astype(np.float32)
                    else:
                        points_rgb = np.full((points_xyz.shape[0], 3), 128.0, dtype=np.float32)

                    if points_xyz.shape[0] > 0:
                        points_xyzrgb = np.hstack((points_xyz, points_rgb))
                        num_curr = points_xyzrgb.shape[0]

                        if num_curr >= NUM_POINTS_EXPECTED:
                            indices = np.random.choice(num_curr, NUM_POINTS_EXPECTED, replace=False)
                            sampled_points_np = points_xyzrgb[indices]
                        elif num_curr > 0:
                            padding_indices = np.random.choice(num_curr, NUM_POINTS_EXPECTED - num_curr, replace=True)
                            sampled_points_np = np.vstack((points_xyzrgb, points_xyzrgb[padding_indices]))
                elif should_print_debug_here:
                     print("DEBUG: processed_o3d_pcd was None or had no points from internal_create. sampled_points_np will be zeros.")


            except Exception as e_pcd_proc_main_loop:
                print(f"    !!!!!! 轨迹 {i+1}, 帧 {frame_idx}: 在点云创建(internal)或采样中发生异常: {e_pcd_proc_main_loop} !!!!!!")
                traceback.print_exc()

            is_effectively_zero = np.all(np.abs(sampled_points_np) < 1e-9)

            if should_print_debug_here: # 调试打印（与之前版本相同）
                print(f"DEBUG: sampled_points_np (shape: {sampled_points_np.shape}) after final sampling:")
                if sampled_points_np.size > 0:
                     print(f"DEBUG: sampled_points_np XYZ min: {np.min(sampled_points_np[:,:3], axis=0)}, max: {np.max(sampled_points_np[:,:3], axis=0)}")
                     print(f"DEBUG: sampled_points_np RGB min: {np.min(sampled_points_np[:,3:], axis=0)}, max: {np.max(sampled_points_np[:,3:], axis=0)}")
                else: print("DEBUG: sampled_points_np is empty after sampling.")
                print(f"DEBUG: is_effectively_zero: {is_effectively_zero}")
                first_problematic_frame_debugged_main = True
                if debug_first_frame_only:
                    print("DEBUG: Exiting after first debugged frame due to debug_first_frame_only=True.")
                    # return # 如果只想看一帧的调试输出，可以在这里取消注释 return

            if is_effectively_zero:
                zero_pcd_frames_in_traj += 1; continue

            try: # state 和 action 构建逻辑与之前相同
                joint_pos_np = np.array(frame_data['joint_positions'], dtype=np.float32)
                joint_vel_np = np.array(frame_data['joint_velocities'], dtype=np.float32)
                gripper_state_np = np.array([frame_data['gripper_openness_ratio']], dtype=np.float32)
                obj_pos_rel_list = frame_data['object_position_relative_to_base']
                obj_ori_rel_list_xyzw = frame_data['object_orientation_relative_to_base_xyzw']
                if not (isinstance(obj_ori_rel_list_xyzw, (list, tuple)) and len(obj_ori_rel_list_xyzw) == 4):
                    raise ValueError(f"object_orientation_relative_to_base_xyzw 格式不正确")
                obj_ori_rel_quat_obj = gymapi.Quat(*obj_ori_rel_list_xyzw)
                obj_ori_rel_rpy_list = convert_gymapi_quat_to_rpy_list(obj_ori_rel_quat_obj)
                state_vector = np.concatenate([
                    joint_pos_np, joint_vel_np,
                    np.array(obj_pos_rel_list, dtype=np.float32),
                    np.array(obj_ori_rel_list_xyzw, dtype=np.float32),
                    np.array(obj_ori_rel_rpy_list, dtype=np.float32),
                    gripper_state_np
                ]).astype(np.float32)

                target_eef_p_rel_list = frame_data['eef_target_position_relative_to_base']
                target_eef_r_rel_list_xyzw = frame_data['eef_target_orientation_relative_to_base_xyzw']
                target_gripper_ratio_val = frame_data['gripper_openness_ratio']
                if not (isinstance(target_eef_r_rel_list_xyzw, (list, tuple)) and len(target_eef_r_rel_list_xyzw) == 4):
                     raise ValueError(f"eef_target_orientation_relative_to_base_xyzw 格式不正确")
                target_eef_r_relative_quat_obj = gymapi.Quat(*target_eef_r_rel_list_xyzw)
                target_eef_r_relative_rpy_list = convert_gymapi_quat_to_rpy_list(target_eef_r_relative_quat_obj)
                action_vector = np.concatenate([
                    np.array(target_eef_p_rel_list, dtype=np.float32),
                    np.array(target_eef_r_relative_rpy_list, dtype=np.float32),
                    np.array([target_gripper_ratio_val], dtype=np.float32)
                ]).astype(np.float32)

                current_traj_pc_list.append(sampled_points_np)
                current_traj_states_list.append(state_vector)
                current_traj_actions_list.append(action_vector)
            except KeyError as ke: print(f"     KeyError: {ke} in frame {frame_idx}, skipping frame."); pass
            except ValueError as ve: print(f"    ValueError: {ve} in frame {frame_idx}, skipping frame."); pass
            except Exception as e_state_action: print(f"    Unknown error: {e_state_action} in frame {frame_idx}, skipping frame."); traceback.print_exc(); pass

        # --- 单条轨迹内帧循环结束 ---
        # ... (后续的轨迹级检查、Zarr写入和日志打印逻辑与之前版本相同) ...
        # ... 为了简洁，这里省略了这些代码块，但它们应该保持不变 ...
        valid_frames_collected_in_traj = len(current_traj_pc_list)
        if not (valid_frames_collected_in_traj == len(current_traj_states_list) == len(current_traj_actions_list)):
            print(f"    !!!!!! 严重错误: 轨迹 {i+1} 内部数据列表长度不一致! 跳过此轨迹。")
            skipped_traj_count += 1; # 在单轨迹模式下，如果这里失败，将没有数据写入
            if i == 0: raw_trajectory_data = [] # 确保后续Zarr写入不会发生
        
        if pcd_processing_attempted_in_traj > 0:
            zero_pcd_frame_ratio = zero_pcd_frames_in_traj / pcd_processing_attempted_in_traj
            if zero_pcd_frame_ratio > 0.30: # 对于单轨迹调试，这个比例可能需要调整或特别注意
                print(f"    警告: 轨迹 {i+1} ({os.path.basename(traj_file_path)}) 因点云全零帧过多而被放弃。")
                print(f"      (尝试处理 {pcd_processing_attempted_in_traj} 帧, 其中 {zero_pcd_frames_in_traj} 帧点云为零, "
                      f"比例: {zero_pcd_frame_ratio*100:.2f}%)")
                skipped_traj_count += 1
                if i == 0: raw_trajectory_data = [] 

        if initial_frames_skipped_in_traj > 0 or zero_pcd_frames_in_traj > 0 :
             print(f"    轨迹 {i+1} ({os.path.basename(traj_file_path)}) 统计: 初步跳过 {initial_frames_skipped_in_traj}, 点云处理 {pcd_processing_attempted_in_traj}, 点云为零 {zero_pcd_frames_in_traj}. 最终有效: {valid_frames_collected_in_traj}.")

        if valid_frames_collected_in_traj < MIN_FRAMES_PER_TRAJECTORY:
            if valid_frames_collected_in_traj > 0 or initial_frames_skipped_in_traj > 0 or zero_pcd_frames_in_traj > 0:
                print(f"    轨迹 {i+1}: 有效帧数 {valid_frames_collected_in_traj} < 最低要求 {MIN_FRAMES_PER_TRAJECTORY}，跳过。")
            else: print(f"    轨迹 {i+1}: 没有收集到有效帧数据，跳过。")
            skipped_traj_count += 1
            if i == 0: raw_trajectory_data = []
        
        # 只有当确实有有效数据可写时才继续 (对于单轨迹，如果raw_trajectory_data被清空，这里不会执行)
        if valid_frames_collected_in_traj > 0 and raw_trajectory_data : # 确保raw_trajectory_data未被清空
            traj_pc_np = np.array(current_traj_pc_list, dtype=np.float32)
            traj_state_np = np.array(current_traj_states_list, dtype=np.float32)
            traj_action_np = np.array(current_traj_actions_list, dtype=np.float32)

            if not first_trajectory_successfully_written:
                # 初始化Zarr数据集 (逻辑与之前相同)
                pc_shape = (0, NUM_POINTS_EXPECTED, 6); pc_chunks = (1, NUM_POINTS_EXPECTED, 6)
                pc_dataset = data_group.require_dataset('point_cloud', shape=pc_shape, chunks=pc_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                
                state_dim = traj_state_np.shape[1]; state_shape = (0, state_dim); state_chunks = (min(100, valid_frames_collected_in_traj), state_dim)
                state_dataset = data_group.require_dataset('state', shape=state_shape, chunks=state_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                
                action_dim = traj_action_np.shape[1]; action_shape = (0, action_dim); action_chunks = (min(100, valid_frames_collected_in_traj), action_dim)
                action_dataset = data_group.require_dataset('action', shape=action_shape, chunks=action_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                
                print(f"    Zarr数据集已初始化。State dim: {state_dim}, Action dim: {action_dim}")
                first_trajectory_successfully_written = True
            
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
                # continue # 在单轨迹模式下，这里出错意味着没有数据
            else: # 只有在append成功时才更新
                current_total_frames_in_zarr += valid_frames_collected_in_traj
                episode_ends_list.append(current_total_frames_in_zarr)
                processed_traj_count += 1
                print(f"    轨迹 {i+1} 处理完毕并添加。有效帧数: {valid_frames_collected_in_traj}。当前Zarr总有效帧数: {current_total_frames_in_zarr}")
        
        # #############################################################
        # ############### 修改点: 在第一次迭代后跳出 #####################
        # #############################################################
        print("已处理完第一个（或唯一的）轨迹，将结束主循环。")
        # break # 在 for 循环外层控制，这里不需要了
        # #############################################################
    else: # 如果 trajectory_files 为空
        print("没有找到轨迹文件进行处理。")


    if not first_trajectory_successfully_written:
        print("[主循环] 错误：没有有效轨迹被处理并写入Zarr文件。")
        if os.path.exists(OUTPUT_ZARR_PATH) and os.path.isdir(OUTPUT_ZARR_PATH):
            try:
                if not any(os.scandir(OUTPUT_ZARR_PATH)):
                    shutil.rmtree(OUTPUT_ZARR_PATH)
                    print(f"  已删除空的Zarr目录: {OUTPUT_ZARR_PATH}")
            except Exception as e_del_empty_zarr: print(f"  删除空Zarr目录时出错: {e_del_empty_zarr}")
        return

    if 'episode_ends' in meta_group: del meta_group['episode_ends']
    if episode_ends_list:
        meta_group.create_dataset('episode_ends', data=np.array(episode_ends_list, dtype=np.int64), chunks=(min(1000, len(episode_ends_list)),))
    else: print("警告: episode_ends_list 为空，没有元数据被保存到 meta/episode_ends。")

    # ... (脚本末尾的日志打印与之前相同，此处省略以保持简洁) ...
    print(f"\n[计时] Zarr文件处理主循环耗时: {time.time() - main_loop_start_time:.2f} 秒")
    print("\n数据转换完成。")
    print(f"  总轨迹文件数 (尝试处理的): {1 if len(trajectory_files) > 0 else 0}")
    print(f"  成功处理并写入Zarr的轨迹数: {processed_traj_count}")
    print(f"  跳过的轨迹数 (因各种原因): {skipped_traj_count}") # 这个在单轨迹模式下可能一直是0或1
    print(f"  Zarr中总帧数: {current_total_frames_in_zarr}")

    if pc_dataset is not None : print(f"点云数据集最终形状: {pc_dataset.shape}, 分块: {pc_dataset.chunks}")
    if state_dataset is not None : print(f"状态数据集最终形状: {state_dataset.shape}, 分块: {state_dataset.chunks}")
    if action_dataset is not None : print(f"动作数据集最终形状: {action_dataset.shape}, 分块: {action_dataset.chunks}")
    
    if episode_ends_list and 'episode_ends' in meta_group and meta_group['episode_ends'].size > 0:
        print(f"片段结束点 ({len(episode_ends_list)}条轨迹) 最后5条: {meta_group['episode_ends'][-5:]}")
    elif episode_ends_list: print(f"片段结束点 ({len(episode_ends_list)}条轨迹) 最后5条: {np.array(episode_ends_list, dtype=np.int64)[-5:]} (元数据中可能未正确写入或为空)")
    else: print("片段结束点列表为空。")

    print(f"\n[计时] 脚本总运行耗时: {time.time() - overall_start_time:.2f} 秒")
    if first_trajectory_successfully_written:
        print(f"\n数据成功写入Zarr文件: {OUTPUT_ZARR_PATH}");
        print("Zarr文件结构 (顶层):");
        try:
            for item_name in zarr_root.keys():
                item_obj = zarr_root[item_name]
                print(f"  - {'Group' if isinstance(item_obj, zarr.hierarchy.Group) else 'Array'}: {item_name}")
                if isinstance(item_obj, zarr.hierarchy.Group):
                    for sub_item_name in item_obj.keys():
                        sub_item_obj = item_obj[sub_item_name]
                        print(f"    - {'Group' if isinstance(sub_item_obj, zarr.hierarchy.Group) else 'Array'}: {sub_item_name}, Shape: {getattr(sub_item_obj, 'shape', 'N/A')}, Chunks: {getattr(sub_item_obj, 'chunks', 'N/A')}, Dtype: {getattr(sub_item_obj, 'dtype', 'N/A')}")
        except Exception as e_zarr_tree: print(f"打印Zarr树信息时出错: {e_zarr_tree}")
    else: print(f"没有数据成功写入Zarr文件: {OUTPUT_ZARR_PATH}")


if __name__ == '__main__':
    main()