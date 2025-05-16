import json
import os
import glob
import numpy as np
import zarr
from isaacgym import gymapi # 假设此库可用
import open3d as o3d
from typing import Tuple, Dict, Union
from scipy.spatial.transform import Rotation as R_scipy
import shutil
import torch
from ultralytics import FastSAM
import cv2
import traceback
import time # <--- 导入time模块

from model.point_cloud_crop import segment_and_crop_with_fastsam as external_point_cloud_processor
from model.point_cloud_crop import create_point_cloud_from_depth_internal
from model.point_cloud_crop import save_point_cloud_to_html

NUM_POINTS_EXPECTED = 16384*16
OUTPUT_ZARR_PATH = '/data1/zhoufang/dataset_output/processed_data_fastsam_hand_object_only_v3.zarr'
DATA_DIR = '/data1/zhoufang/dataset/collected_data'
MIN_FRAMES_PER_TRAJECTORY = 50
IMAGE_WIDTH_CONFIG = 640
IMAGE_HEIGHT_CONFIG = 480
FASTSAM_MODEL_PATH = './model/FastSAM-x.pt'
TEXT_PROMPT_HAND = "robotic gripper with fingers"
TEXT_PROMPT_OBJECT = "small white cube object on the brown table"
DEPTH_SCALE = 1.0
print(f"重要提示: DEPTH_SCALE 设置为 {DEPTH_SCALE}。这假设您的原始深度数据单位是米。")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 相机与机器人姿态设置 ---
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
    overall_start_time = time.time() # 总开始时间
    print(f"将使用图像尺寸: Width={IMAGE_WIDTH_CONFIG}, Height={IMAGE_HEIGHT_CONFIG}")
    print(f"处理点云将使用设备: {DEVICE}")

    t_start_load_model = time.time()
    print(f"正在从 {FASTSAM_MODEL_PATH} 加载 FastSAM 模型...")
    if not os.path.exists(FASTSAM_MODEL_PATH):
        print(f"错误: FastSAM 模型未找到: {FASTSAM_MODEL_PATH}"); return
    try:
        fastsam_model_instance = FastSAM(FASTSAM_MODEL_PATH)
        print(f"FastSAM 模型已加载。耗时: {time.time() - t_start_load_model:.4f} 秒")
    except Exception as e:
        print(f"错误: 无法加载FastSAM模型: {e}"); traceback.print_exc(); return

    t_start_find_files = time.time()
    trajectory_files = []
    abs_data_dir = os.path.abspath(DATA_DIR)
    if not os.path.isdir(abs_data_dir):
        print(f"错误: 数据目录 '{abs_data_dir}' 不存在。"); return
    for root_dir, _, files in os.walk(abs_data_dir):
        for filename in files:
            if filename.startswith('trajectory_') and filename.endswith('.json'):
                trajectory_files.append(os.path.join(root_dir, filename))

    trajectory_files = sorted(trajectory_files)
    if not trajectory_files:
        print(f"错误：在'{abs_data_dir}'中没有找到轨迹文件"); return
    print(f"找到 {len(trajectory_files)} 条轨迹文件。耗时: {time.time() - t_start_find_files:.4f} 秒")
    print(f"将使用以下文本提示: 手='{TEXT_PROMPT_HAND}', 物体='{TEXT_PROMPT_OBJECT}' (仅保留手和物体)")

    debug_first_frame_only = True # 是否仅调试第一帧
    debug_fastsam_conf_hand = 0.35
    debug_fastsam_conf_object = 0.15

    if trajectory_files and debug_first_frame_only:
        debug_overall_start_time = time.time() # <--- 调试块开始时间

        first_traj_file_path = trajectory_files[0]
        print(f"\n[调试模式] 正在处理第一条轨迹的第一有效帧以生成HTML: {first_traj_file_path}")
        print(f"  调试用FastSAM置信度: 手={debug_fastsam_conf_hand}, 物体={debug_fastsam_conf_object}")
        current_traj_dir_debug = os.path.dirname(first_traj_file_path)
        first_valid_frame_data_debug = None
        
        t_start_load_json = time.time()
        try:
            with open(first_traj_file_path, 'r') as f_debug:
                raw_traj_data_debug = json.load(f_debug)
            if isinstance(raw_traj_data_debug, list) and len(raw_traj_data_debug) > 0:
                for frame_idx_debug_loop, frame_data_debug_loop in enumerate(raw_traj_data_debug):
                    required_keys_check = ['depth_image_path', 'rgb_image_path', 'camera_K_matrix']
                    if all(key in frame_data_debug_loop and frame_data_debug_loop[key] is not None for key in required_keys_check):
                        first_valid_frame_data_debug = frame_data_debug_loop
                        print(f"  找到有效帧 (原始索引 {frame_idx_debug_loop})。"); break
                if not first_valid_frame_data_debug: print(f"  警告: 轨迹 {first_traj_file_path} 中无有效帧数据。")
            else: print(f"  警告: 轨迹 {first_traj_file_path} 为空或格式不正确。")
        except Exception as e: print(f"  错误: 加载调试轨迹时出错: {e}"); traceback.print_exc()
        print(f"  [计时] 加载轨迹JSON耗时: {time.time() - t_start_load_json:.4f} 秒")

        if first_valid_frame_data_debug:
            try:
                t_frame_processing_start = time.time() # <--- 单帧处理开始时间

                k_matrix_debug = np.array(first_valid_frame_data_debug['camera_K_matrix'], dtype=np.float32)
                depth_path_debug = os.path.join(current_traj_dir_debug, first_valid_frame_data_debug['depth_image_path'])
                rgb_path_debug = os.path.join(current_traj_dir_debug, first_valid_frame_data_debug['rgb_image_path'])
                
                t_start_load_images = time.time()
                depth_np_debug = np.load(depth_path_debug)
                rgb_np_debug_loaded = np.load(rgb_path_debug)
                print(f"  [计时] 加载RGB和深度图像耗时: {time.time() - t_start_load_images:.4f} 秒")

                t_start_preprocess_images = time.time()
                if not np.issubdtype(depth_np_debug.dtype, np.floating):
                    depth_np_debug = depth_np_debug.astype(np.float32)
                rgb_input_debug = rgb_np_debug_loaded
                prepared_rgb_for_debug = rgb_input_debug.copy()
                if not np.issubdtype(prepared_rgb_for_debug.dtype, np.uint8):
                    if np.issubdtype(prepared_rgb_for_debug.dtype, np.floating) and prepared_rgb_for_debug.max()<=1.0 and prepared_rgb_for_debug.min()>=0.0:
                        prepared_rgb_for_debug=(prepared_rgb_for_debug*255).astype(np.uint8)
                    else: prepared_rgb_for_debug=np.clip(prepared_rgb_for_debug,0,255).astype(np.uint8)

                if prepared_rgb_for_debug.ndim == 3 and prepared_rgb_for_debug.shape[2] == 4:
                    rgb_bgr_debug = cv2.cvtColor(prepared_rgb_for_debug, cv2.COLOR_RGBA2BGR)
                elif prepared_rgb_for_debug.ndim == 3 and prepared_rgb_for_debug.shape[2] == 3:
                    rgb_bgr_debug = prepared_rgb_for_debug
                elif prepared_rgb_for_debug.ndim == 2:
                    rgb_bgr_debug = cv2.cvtColor(prepared_rgb_for_debug, cv2.COLOR_GRAY2BGR)
                else: rgb_bgr_debug = None
                print(f"  [计时] 图像预处理耗时: {time.time() - t_start_preprocess_images:.4f} 秒")

                if rgb_bgr_debug is not None and \
                   rgb_bgr_debug.shape[0] == IMAGE_HEIGHT_CONFIG and rgb_bgr_debug.shape[1] == IMAGE_WIDTH_CONFIG and \
                   depth_np_debug.shape[0] == IMAGE_HEIGHT_CONFIG and depth_np_debug.shape[1] == IMAGE_WIDTH_CONFIG :
                    
                    print("  [调试模式] 1. 生成并保存原始点云 ...")
                    t_start_raw_pcd = time.time()
                    try:
                        raw_pcd_check, _ = create_point_cloud_from_depth_internal(
                            depth_np_debug, k_matrix_debug, DEPTH_SCALE, rgb_image=rgb_bgr_debug,
                            image_height_config=IMAGE_HEIGHT_CONFIG, image_width_config=IMAGE_WIDTH_CONFIG
                        )
                        print(f"    [计时] create_point_cloud_from_depth_internal 耗时: {time.time() - t_start_raw_pcd:.4f} 秒")
                        if raw_pcd_check.has_points():
                            t_start_save_raw_html = time.time()
                            save_point_cloud_to_html(
                                {'raw_colored_pcd_internal_func': raw_pcd_check},
                                './debug_first_frame_raw_colored_pcd_internal_func.html',
                                title="Debug: Raw Colored PCD (Our Internal Func)"
                            )
                            print(f"    [计时] 保存原始点云HTML耗时: {time.time() - t_start_save_raw_html:.4f} 秒")
                    except Exception as e_html_raw:
                        print(f"    [调试模式] 生成原始点云HTML时出错: {e_html_raw}")

                    print("  [调试模式] 2. 生成并保存处理后的点云 (仅手和物体)...")
                    debug_ground_threshold = None
                    
                    t_start_external_proc = time.time()
                    processed_data, debug_segmentation_images = external_point_cloud_processor(
                        rgb_image_input=rgb_bgr_debug, depth_image_input=depth_np_debug,
                        camera_k_matrix=k_matrix_debug, depth_processing_scale=DEPTH_SCALE,
                        fastsam_model_instance=fastsam_model_instance,
                        text_prompt_hand=TEXT_PROMPT_HAND, text_prompt_object=TEXT_PROMPT_OBJECT,
                        device_for_fastsam=DEVICE, return_separate_segments=True,
                        image_width_config_for_fastsam=IMAGE_WIDTH_CONFIG, image_height_config_for_fastsam=IMAGE_HEIGHT_CONFIG,
                        ground_removal_z_threshold=debug_ground_threshold,
                        return_debug_images=True,
                        fastsam_conf_hand=debug_fastsam_conf_hand,
                        fastsam_conf_object=debug_fastsam_conf_object
                    )
                    print(f"    [计时] external_point_cloud_processor (FastSAM分割裁剪) 耗时: {time.time() - t_start_external_proc:.4f} 秒")
                    processed_pcds_dict_debug = processed_data

                    t_start_save_masks = time.time()
                    if debug_segmentation_images:
                        print("  [调试模式] 保存FastSAM 2D分割掩码图像...")
                        for img_name, img_data in debug_segmentation_images.items():
                            if img_data is not None and img_data.size > 0 :
                                save_path = f"./debug_fastsam_{img_name}.png"
                                try: cv2.imwrite(save_path, img_data); print(f"    已保存: {save_path}")
                                except Exception as e_save_mask: print(f"    保存调试掩码图像 {save_path} 失败: {e_save_mask}")
                            else: print(f"    跳过保存空的调试掩码图像: {img_name}")
                    print(f"    [计时] 保存FastSAM 2D掩码图像耗时: {time.time() - t_start_save_masks:.4f} 秒")
                    
                    t_start_save_processed_html = time.time()
                    if processed_pcds_dict_debug and isinstance(processed_pcds_dict_debug, dict):
                        # ... (打印3D分割部分信息)
                        combined_pcd_debug = processed_pcds_dict_debug.get('combined', o3d.geometry.PointCloud())
                        hand_pcd_segment = processed_pcds_dict_debug.get('hand', o3d.geometry.PointCloud()) # 获取单独的手和物体
                        object_pcd_segment = processed_pcds_dict_debug.get('object', o3d.geometry.PointCloud())

                        if combined_pcd_debug.has_points() or hand_pcd_segment.has_points() or object_pcd_segment.has_points():
                            save_point_cloud_to_html(
                                processed_pcds_dict_debug,
                                './debug_first_frame_processed_hand_object_only.html',
                                title="Debug: Processed (Hand & Object Only, Orig. Colors)"
                            )
                        else: print("    [调试模式] 警告: 处理后的手、物体及组合点云均为空。")
                    else: print("    [调试模式] 警告: 点云分割返回为空或格式不正确。")
                    print(f"    [计时] 保存处理后点云HTML耗时: {time.time() - t_start_save_processed_html:.4f} 秒")
                else:
                    # ... (处理图像格式或尺寸不符的情况)
                    if rgb_bgr_debug is None: print(f"    [调试模式] 警告: RGB图像未能预处理，跳过点云生成。")
                    else: print(f"    [调试模式] 警告: RGB或深度图像形状不符合预期，跳过点云生成。")


                print(f"  [计时] 单帧总处理耗时: {time.time() - t_frame_processing_start:.4f} 秒")
                print("[调试模式] 第一帧点云HTML生成尝试完毕。")
                if debug_first_frame_only:
                    print(f"[计时] 调试块总耗时: {time.time() - debug_overall_start_time:.4f} 秒")
                    print("[调试模式] 'debug_first_frame_only' 为 True, 脚本现在退出。"); return
            except FileNotFoundError as e: print(f"  [调试模式] 错误: 未找到图像文件: {e}")
            except Exception as e: print(f"  [调试模式] 处理第一帧时发生未知错误: {e}"); traceback.print_exc()
            if debug_first_frame_only: return
    # ----- 结束：HTML调试 -----

    # Zarr 文件处理主循环 (这部分通常不需要详细计时，除非整个过程非常慢)
    main_loop_start_time = time.time()

    if os.path.exists(OUTPUT_ZARR_PATH):
        # ... (删除旧Zarr文件)
        pass

    zarr_root = zarr.open_group(OUTPUT_ZARR_PATH, mode='a')
    # ... (Zarr组创建)

    main_loop_conf_hand = 0.25 # 使用您最终确定的值
    main_loop_conf_object = 0.25 # 使用您最终确定的值
    print(f"主循环FastSAM置信度: 手={main_loop_conf_hand}, 物体={main_loop_conf_object}")

    print(f"\n开始处理全部 {len(trajectory_files)} 条轨迹...")
    # ... (主循环代码与之前版本一致，这里不再重复粘贴以节省空间)
    # 您可以在主循环内部的关键步骤（如 external_point_cloud_processor 调用前后）也加入计时
    # 但要注意，过于频繁的 time.time() 调用本身也会带来微小的开销

    # --- 主循环代码开始 (与上一版本相同) ---
    data_group = zarr_root.require_group('data')
    meta_group = zarr_root.require_group('meta')
    episode_ends_list = []
    current_total_frames = 0
    pc_dataset, state_dataset, action_dataset = None, None, None
    first_trajectory_processed_successfully = False

    for i, traj_file_path in enumerate(trajectory_files):
        # ... (轨迹加载等与之前一致)
        # print(f"  处理轨迹 {i+1}/{len(trajectory_files)}: {os.path.basename(traj_file_path)}")
        current_traj_dir = os.path.dirname(traj_file_path)
        try:
            with open(traj_file_path, 'r') as f: raw_trajectory_data = json.load(f)
            if not isinstance(raw_trajectory_data, list) or not raw_trajectory_data: continue
        except Exception: continue
        current_traj_pc, current_traj_states, current_traj_actions = [], [], []
        skipped_frames_count = 0

        for frame_idx, frame_data in enumerate(raw_trajectory_data):
            # ... (帧数据校验、图像加载和预处理等与之前一致)
            required_keys = {
                'depth_image_path': str, 'rgb_image_path': str, 'camera_K_matrix': list,
                'joint_positions': list, 'joint_velocities': list,
                'eef_target_position_relative_to_base': list,
                'eef_target_orientation_relative_to_base_xyzw': list,
                'gripper_openness_ratio': (float, int),
                'object_position_relative_to_base': list,
                'object_orientation_relative_to_base_xyzw': list
            }
            missing_or_invalid_keys = []
            for key, exp_type in required_keys.items():
                if key not in frame_data or frame_data[key] is None: missing_or_invalid_keys.append(f"{key}(None)")
                elif not isinstance(frame_data[key], exp_type): missing_or_invalid_keys.append(f"{key}(type)")
            if missing_or_invalid_keys: skipped_frames_count += 1; continue
            k_matrix = np.array(frame_data['camera_K_matrix'], dtype=np.float32)
            if k_matrix.shape != (3,3): skipped_frames_count += 1; continue
            depth_path = os.path.join(current_traj_dir, frame_data['depth_image_path'])
            rgb_path = os.path.join(current_traj_dir, frame_data['rgb_image_path'])
            try:
                depth_np = np.load(depth_path); rgb_np_loaded = np.load(rgb_path)
                if rgb_np_loaded.shape[0]!=IMAGE_HEIGHT_CONFIG or rgb_np_loaded.shape[1]!=IMAGE_WIDTH_CONFIG or \
                   depth_np.shape[0]!=IMAGE_HEIGHT_CONFIG or depth_np.shape[1]!=IMAGE_WIDTH_CONFIG:
                    skipped_frames_count += 1; continue
                if not np.issubdtype(depth_np.dtype, np.floating): depth_np = depth_np.astype(np.float32)
            except FileNotFoundError: skipped_frames_count += 1; continue
            except Exception: skipped_frames_count += 1; continue
            rgb_input_main = rgb_np_loaded.copy()
            if not np.issubdtype(rgb_input_main.dtype, np.uint8):
                if np.issubdtype(rgb_input_main.dtype, np.floating) and rgb_input_main.max()<=1.0 and rgb_input_main.min()>=0.0:
                    rgb_input_main=(rgb_input_main*255).astype(np.uint8)
                else: rgb_input_main=np.clip(rgb_input_main,0,255).astype(np.uint8)
            if rgb_input_main.ndim == 3 and rgb_input_main.shape[2] == 4: rgb_bgr_main = cv2.cvtColor(rgb_input_main, cv2.COLOR_RGBA2BGR)
            elif rgb_input_main.ndim == 3 and rgb_input_main.shape[2] == 3: rgb_bgr_main = rgb_input_main
            elif rgb_input_main.ndim == 2: rgb_bgr_main = cv2.cvtColor(rgb_input_main, cv2.COLOR_GRAY2BGR)
            else: skipped_frames_count += 1; continue
            sampled_points_np = np.zeros((NUM_POINTS_EXPECTED, 6), dtype=np.float32)
            if not (rgb_bgr_main.ndim == 3 and rgb_bgr_main.shape[2] == 3): skipped_frames_count += 1; continue

            try:
                active_ground_threshold = None
                processed_o3d_pcd = external_point_cloud_processor(
                    rgb_image_input=rgb_bgr_main,
                    depth_image_input=depth_np, camera_k_matrix=k_matrix,
                    depth_processing_scale=DEPTH_SCALE,
                    fastsam_model_instance=fastsam_model_instance,
                    text_prompt_hand=TEXT_PROMPT_HAND,
                    text_prompt_object=TEXT_PROMPT_OBJECT,
                    device_for_fastsam=DEVICE, return_separate_segments=False,
                    image_width_config_for_fastsam=IMAGE_WIDTH_CONFIG,
                    image_height_config_for_fastsam=IMAGE_HEIGHT_CONFIG,
                    ground_removal_z_threshold=active_ground_threshold,
                    return_debug_images=False,
                    fastsam_conf_hand=main_loop_conf_hand,
                    fastsam_conf_object=main_loop_conf_object
                )
                if processed_o3d_pcd and processed_o3d_pcd.has_points():
                    points_xyz = np.asarray(processed_o3d_pcd.points, dtype=np.float32)
                    if processed_o3d_pcd.has_colors(): points_rgb = np.asarray(processed_o3d_pcd.colors, dtype=np.float32)
                    else: points_rgb = np.full((points_xyz.shape[0], 3), 0.5, dtype=np.float32)
                    if points_xyz.shape[0] > 0:
                        points_xyzrgb = np.hstack((points_xyz, points_rgb))
                        num_curr = points_xyzrgb.shape[0]
                        if num_curr > NUM_POINTS_EXPECTED:
                            indices = np.random.choice(num_curr, NUM_POINTS_EXPECTED, replace=False)
                            sampled_points_np = points_xyzrgb[indices]
                        elif num_curr < NUM_POINTS_EXPECTED and num_curr > 0:
                            padding_indices = np.random.choice(num_curr, NUM_POINTS_EXPECTED - num_curr, replace=True)
                            sampled_points_np = np.vstack((points_xyzrgb, points_xyzrgb[padding_indices]))
                        elif num_curr == NUM_POINTS_EXPECTED: sampled_points_np = points_xyzrgb
            except Exception: pass
            current_traj_pc.append(sampled_points_np)
            joint_pos_np=np.array(frame_data['joint_positions'],dtype=np.float32);joint_vel_np=np.array(frame_data['joint_velocities'],dtype=np.float32)
            gripper_state_np=np.array([frame_data['gripper_openness_ratio']],dtype=np.float32)
            obj_pos_rel_list=frame_data['object_position_relative_to_base'];obj_ori_rel_list_xyzw=frame_data['object_orientation_relative_to_base_xyzw']
            obj_ori_rel_quat_obj=gymapi.Quat(*obj_ori_rel_list_xyzw);obj_ori_rel_rpy_list=convert_gymapi_quat_to_rpy_list(obj_ori_rel_quat_obj)
            state_vector=np.concatenate([joint_pos_np,joint_vel_np,np.array(obj_pos_rel_list,dtype=np.float32),np.array(obj_ori_rel_list_xyzw,dtype=np.float32),np.array(obj_ori_rel_rpy_list,dtype=np.float32),gripper_state_np]).astype(np.float32)
            current_traj_states.append(state_vector)
            target_eef_p_rel_list=frame_data['eef_target_position_relative_to_base'];target_eef_r_rel_list_xyzw=frame_data['eef_target_orientation_relative_to_base_xyzw']
            target_gripper_ratio_val=frame_data['gripper_openness_ratio']
            target_eef_r_relative_quat_obj=gymapi.Quat(*target_eef_r_rel_list_xyzw);target_eef_r_relative_rpy_list=convert_gymapi_quat_to_rpy_list(target_eef_r_relative_quat_obj)
            action_vector=np.concatenate([np.array(target_eef_p_rel_list,dtype=np.float32),np.array(target_eef_r_rel_list_xyzw,dtype=np.float32),np.array(target_eef_r_relative_rpy_list,dtype=np.float32),np.array([target_gripper_ratio_val],dtype=np.float32)]).astype(np.float32)
            current_traj_actions.append(action_vector)

        num_valid_frames_collected = len(current_traj_pc)
        if skipped_frames_count > 0: print(f"    轨迹 {i+1}: 跳过了 {skipped_frames_count} 帧, 收集了 {num_valid_frames_collected} 有效帧。")
        if num_valid_frames_collected < MIN_FRAMES_PER_TRAJECTORY: continue
        if num_valid_frames_collected > 0:
            traj_pc_np = np.array(current_traj_pc, dtype=np.float32); traj_state_np = np.array(current_traj_states, dtype=np.float32); traj_action_np = np.array(current_traj_actions, dtype=np.float32)
            if not first_trajectory_processed_successfully:
                pc_shape = (0, NUM_POINTS_EXPECTED, 6); pc_chunks = (1, NUM_POINTS_EXPECTED, 6)
                pc_dataset = data_group.require_dataset('point_cloud', shape=pc_shape, chunks=pc_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                state_dim = traj_state_np.shape[1]; state_shape = (0, state_dim); state_chunks = (100, state_dim)
                state_dataset = data_group.require_dataset('state', shape=state_shape, chunks=state_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                action_dim = traj_action_np.shape[1]; action_shape = (0, action_dim); action_chunks = (100, action_dim)
                action_dataset = data_group.require_dataset('action', shape=action_shape, chunks=action_chunks, dtype='float32', exact=False, compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                first_trajectory_processed_successfully = True
            pc_dataset.append(traj_pc_np); state_dataset.append(traj_state_np); action_dataset.append(traj_action_np)
            current_total_frames += num_valid_frames_collected; episode_ends_list.append(current_total_frames - 1)
            if (i+1) % 10 == 0 or i == len(trajectory_files) -1 : # 每10条轨迹或最后一条轨迹打印一次进度
                 print(f"  轨迹 {i+1} 有效处理并追加了 {num_valid_frames_collected} 帧。总帧数: {current_total_frames}")
    # --- 主循环代码结束 ---


    if not first_trajectory_processed_successfully:
        print("错误：没有有效轨迹被处理。无法创建Zarr文件。")
        # ... (清理空Zarr目录)
        return

    if 'episode_ends' in meta_group: del meta_group['episode_ends']
    if episode_ends_list:
        meta_group.create_dataset('episode_ends', data=np.array(episode_ends_list, dtype=np.int64))

    print(f"\n[计时] Zarr文件处理主循环耗时: {time.time() - main_loop_start_time:.4f} 秒")
    print("\n数据转换完成。")
    # ... (打印Zarr信息)
    if pc_dataset is not None:print(f"点云形状: {pc_dataset.shape}")
    # ...

    print(f"\n[计时] 脚本总运行耗时: {time.time() - overall_start_time:.4f} 秒")
    print(f"\n数据写入Zarr文件: {OUTPUT_ZARR_PATH}");print("Zarr文件结构:");print(zarr_root.tree())


if __name__ == '__main__':
    try: import cv2
    except ImportError: print("警告: OpenCV (cv2) 未安装。")
    main()