import open3d as o3d
import numpy as np
import cv2
from ultralytics import FastSAM # 假设此为正确的导入
import torch
import plotly.graph_objects as go
import plotly.io as pio
import time # 确保导入 time 模块
import traceback # 虽然在这些函数中没直接用，但保持导入以防万一

def create_point_cloud_from_depth_internal(depth_image, k_matrix, depth_scale_factor, rgb_image=None, image_height_config=480, image_width_config=640):
    fx = k_matrix[0, 0]; fy = k_matrix[1, 1]; cx = k_matrix[0, 2]; cy = k_matrix[1, 2]
    if fx == 0 or fy == 0: return o3d.geometry.PointCloud(), np.array([])
    if depth_scale_factor == 0: return o3d.geometry.PointCloud(), np.array([])

    height, width = depth_image.shape
    if rgb_image is not None:
        if rgb_image.shape[0] != height or rgb_image.shape[1] != width:
            rgb_image_for_color = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            rgb_image_for_color = rgb_image
    else:
        rgb_image_for_color = None

    u_coords = np.arange(width); v_coords = np.arange(height)
    u, v = np.meshgrid(u_coords, v_coords)

    depth_image_float = depth_image.astype(np.float32)
    depth_abs = np.abs(depth_image_float)
    depth_in_meters = depth_abs / depth_scale_factor

    min_valid_depth_m = 0.1
    max_valid_depth_m = 3.0

    processed_depth_for_masking = np.copy(depth_in_meters)
    processed_depth_for_masking[processed_depth_for_masking < min_valid_depth_m] = 0
    processed_depth_for_masking[processed_depth_for_masking > max_valid_depth_m] = 0
    valid_mask = processed_depth_for_masking > 1e-6

    if np.sum(valid_mask) == 0:
        return o3d.geometry.PointCloud(), np.array([])

    z_values_for_calc = depth_in_meters[valid_mask]
    x = (u[valid_mask] - cx) * z_values_for_calc / fx
    y = (v[valid_mask] - cy) * z_values_for_calc / fy
    points_3d = np.vstack((x, y, z_values_for_calc)).T
    pcd = o3d.geometry.PointCloud()
    pixel_map = np.array([])

    if points_3d.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pixel_map = np.vstack((u[valid_mask].flatten(), v[valid_mask].flatten())).T.astype(int)

        if rgb_image_for_color is not None and pixel_map.size > 0:
            try:
                processed_rgb_for_color = rgb_image_for_color.copy()
                if not np.issubdtype(processed_rgb_for_color.dtype, np.uint8):
                    if np.issubdtype(processed_rgb_for_color.dtype, np.floating) and \
                       processed_rgb_for_color.min() >= 0.0 and processed_rgb_for_color.max() <= 1.0:
                        processed_rgb_for_color = (processed_rgb_for_color * 255).astype(np.uint8)
                    else:
                        processed_rgb_for_color = np.clip(processed_rgb_for_color, 0, 255).astype(np.uint8)

                if processed_rgb_for_color.ndim == 3 and processed_rgb_for_color.shape[2] == 4:
                     processed_rgb_for_color = cv2.cvtColor(processed_rgb_for_color, cv2.COLOR_RGBA2RGB)
                elif processed_rgb_for_color.ndim == 3 and processed_rgb_for_color.shape[2] == 3: # BGR or RGB
                    processed_rgb_for_color = cv2.cvtColor(processed_rgb_for_color, cv2.COLOR_BGR2RGB)
                elif processed_rgb_for_color.ndim == 2: # Grayscale
                    processed_rgb_for_color = cv2.cvtColor(processed_rgb_for_color, cv2.COLOR_GRAY2RGB)

                colors_for_pcd = np.zeros((pixel_map.shape[0], 3), dtype=np.float64)
                valid_map_indices = (pixel_map[:, 1] < processed_rgb_for_color.shape[0]) & \
                                    (pixel_map[:, 0] < processed_rgb_for_color.shape[1]) & \
                                    (pixel_map[:, 1] >= 0) & (pixel_map[:, 0] >= 0)

                if np.any(valid_map_indices):
                    valid_pixel_coords_y = pixel_map[valid_map_indices, 1]
                    valid_pixel_coords_x = pixel_map[valid_map_indices, 0]
                    extracted_colors = processed_rgb_for_color[valid_pixel_coords_y, valid_pixel_coords_x]
                    if np.issubdtype(extracted_colors.dtype, np.uint8):
                        extracted_colors = extracted_colors / 255.0
                    if extracted_colors.ndim == 1 and extracted_colors.shape[0] == colors_for_pcd[valid_map_indices].shape[0]:
                        colors_for_pcd[valid_map_indices] = np.tile(extracted_colors[:, np.newaxis], (1, 3))
                    elif extracted_colors.ndim ==2 and extracted_colors.shape[1] == 3:
                        colors_for_pcd[valid_map_indices] = extracted_colors
                    pcd.colors = o3d.utility.Vector3dVector(colors_for_pcd)
            except Exception:
                # print(f"    [create_point_cloud_from_depth_internal] 警告: 上色时出错. 点云将无颜色.")
                pass
    return pcd, pixel_map


def _generate_plotly_trace_from_o3d_pcd(o3d_pcd, marker_color=None, marker_size=2, name=None):
    if not o3d_pcd.has_points(): return None
    points_np = np.asarray(o3d_pcd.points)
    plotly_colors = None
    if o3d_pcd.has_colors(): plotly_colors = np.asarray(o3d_pcd.colors)
    elif marker_color is not None: plotly_colors = marker_color
    return go.Scatter3d(x=points_np[:, 0], y=points_np[:, 1], z=points_np[:, 2], mode='markers', marker=dict(size=marker_size, color=plotly_colors, opacity=0.8), name=name)

def save_point_cloud_to_html(point_clouds_dict, file_path, title="Point Cloud Visualization"):
    traces = []
    if isinstance(point_clouds_dict, o3d.geometry.PointCloud):
        trace = _generate_plotly_trace_from_o3d_pcd(point_clouds_dict, name="Point Cloud")
        if trace: traces.append(trace)
    elif isinstance(point_clouds_dict, dict):
        for name, pcd_item in point_clouds_dict.items():
            default_color_override = None
            if not pcd_item.has_colors():
                if name == 'table': default_color_override = 'rgb(0,165,237)'
                elif name == 'hand': default_color_override = 'rgb(255,0,0)'
                elif name == 'object': default_color_override = 'rgb(0,255,0)'
                elif name == 'combined': default_color_override = 'rgb(128,128,128)'
                elif 'raw_depth' in name or 'raw_colored_pcd' in name : default_color_override = 'rgb(200,200,200)'
                else: default_color_override = 'rgb(128,128,128)'
            trace = _generate_plotly_trace_from_o3d_pcd(pcd_item, marker_color=default_color_override, name=name)
            if trace: traces.append(trace)
    else:
        return

    if not traces:
        fig = go.Figure(layout=go.Layout(title=f"{title} (无数据)", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')))
        try:
            pio.write_html(fig, file_path, auto_open=False, full_html=True)
        except Exception:
            pass
        return

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(showbackground=False, title='X (相机坐标系)'),
            yaxis=dict(showbackground=False, title='Y (相机坐标系)'),
            zaxis=dict(showbackground=False, title='Z (相机坐标系)'),
            bgcolor='white',
            aspectmode='data'
        ),
        margin=dict(l=10, r=10, b=10, t=40)
    )
    fig = go.Figure(data=traces, layout=layout)
    try:
        pio.write_html(fig, file_path, auto_open=False, full_html=True)
        print(f"点云可视化已保存到HTML文件: {file_path}")
    except Exception:
        pass

def get_fastsam_masks_from_text(text_prompt, image_for_model, model_instance, device_to_use, image_size_for_fastsam, conf_threshold=0.25, parent_timing_prefix=""):
    if not text_prompt: return np.zeros(image_for_model.shape[:2], dtype=bool)
    try:
        img_bgr_contiguous = image_for_model

        t_start_predict = time.time()
        results = model_instance.predict(source=img_bgr_contiguous, conf=conf_threshold, texts=[text_prompt], device=device_to_use, retina_masks=True, imgsz=image_size_for_fastsam, verbose=False)
        t_end_predict = time.time()
        # print(f"{parent_timing_prefix}    [子计时] FastSAM predict ('{text_prompt}') 耗时: {t_end_predict - t_start_predict:.4f} 秒")

        if results and results[0].masks is not None:
            all_masks_for_prompt = results[0].masks.data.cpu().numpy().astype(bool)
            if all_masks_for_prompt.ndim == 3 and all_masks_for_prompt.shape[0] > 0: return np.any(all_masks_for_prompt, axis=0)
            elif all_masks_for_prompt.ndim == 2: return all_masks_for_prompt
        return np.zeros(image_for_model.shape[:2], dtype=bool)
    except Exception as e:
        # print(f"FastSAM使用文本提示 '{text_prompt}' 进行推理时发生未知错误: {e}")
        return np.zeros(image_for_model.shape[:2], dtype=bool)

def segment_and_crop_with_fastsam(
    rgb_image_input, depth_image_input, camera_k_matrix, depth_processing_scale,
    fastsam_model_instance, text_prompt_hand=None, text_prompt_object=None, text_prompt_table=None,
    ransac_distance_thresh=0.015, device_for_fastsam=None, return_separate_segments=False,
    image_width_config_for_fastsam=640, image_height_config_for_fastsam=480,
    ground_removal_z_threshold=None,
    return_debug_images=False,
    fastsam_conf_hand=0.25,
    fastsam_conf_object=0.25,
    timing_prefix="  [分割裁剪计时]"
    ):

    overall_func_start_time = time.time()
    debug_images = {}

    
    image_for_processing = rgb_image_input.copy()
    if not np.issubdtype(image_for_processing.dtype, np.uint8):
        if np.issubdtype(image_for_processing.dtype, np.floating) and \
           image_for_processing.min() >= 0.0 and image_for_processing.max() <= 1.0:
            image_for_processing = (image_for_processing * 255).astype(np.uint8)
        else: image_for_processing = np.clip(image_for_processing, 0, 255).astype(np.uint8)

    if image_for_processing.ndim == 3 and image_for_processing.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_for_processing, cv2.COLOR_RGBA2BGR)
    elif image_for_processing.ndim == 3 and image_for_processing.shape[2] == 3:
        image_bgr = image_for_processing
    elif image_for_processing.ndim == 2:
        image_bgr = cv2.cvtColor(image_for_processing, cv2.COLOR_GRAY2BGR)
    else:
        empty_pcd_dict = {'hand': o3d.geometry.PointCloud(), 'object': o3d.geometry.PointCloud(), 'combined': o3d.geometry.PointCloud()}
        if return_separate_segments:
            return (empty_pcd_dict, debug_images) if return_debug_images else empty_pcd_dict
        return (o3d.geometry.PointCloud(), debug_images) if return_debug_images else o3d.geometry.PointCloud()

    image_bgr_contiguous = np.ascontiguousarray(image_bgr)

    if image_bgr_contiguous.shape[0] != image_height_config_for_fastsam or \
       image_bgr_contiguous.shape[1] != image_width_config_for_fastsam:
        image_bgr_for_fastsam_model = cv2.resize(
            image_bgr_contiguous,
            (image_width_config_for_fastsam, image_height_config_for_fastsam),
            interpolation=cv2.INTER_AREA
        )
    else:
        image_bgr_for_fastsam_model = image_bgr_contiguous
    # print(f"{timing_prefix} RGB图像准备和缩放 (FastSAM输入尺寸: {image_width_config_for_fastsam}x{image_height_config_for_fastsam}) 耗时: {time.time() - t_start_prep_rgb:.4f} 秒")


    if device_for_fastsam is None: effective_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: effective_device = device_for_fastsam

    t_start_create_pcd_full = time.time()
    P_full, pcd_pixel_to_point_map = create_point_cloud_from_depth_internal(
        depth_image_input, camera_k_matrix, depth_processing_scale,
        rgb_image=image_bgr,
        image_height_config=depth_image_input.shape[0],
        image_width_config=depth_image_input.shape[1]
    )
    # print(f"{timing_prefix} 初始 P_full (完整点云) 创建 耗时: {time.time() - t_start_create_pcd_full:.4f} 秒")

    empty_pcd_dict_no_table = {'hand': o3d.geometry.PointCloud(), 'object': o3d.geometry.PointCloud(), 'combined': o3d.geometry.PointCloud()}

    if not P_full.has_points():
        if return_separate_segments:
            return (empty_pcd_dict_no_table, debug_images) if return_debug_images else empty_pcd_dict_no_table
        return (o3d.geometry.PointCloud(), debug_images) if return_debug_images else o3d.geometry.PointCloud()

    t_start_ground_removal = time.time()
    if ground_removal_z_threshold is not None:
        points_np_gr = np.asarray(P_full.points)
        colors_np_gr = np.asarray(P_full.colors) if P_full.has_colors() else None
        height_filter_mask_gr = points_np_gr[:, 2] > ground_removal_z_threshold
        if np.sum(height_filter_mask_gr) > 0 :
            P_full.points = o3d.utility.Vector3dVector(points_np_gr[height_filter_mask_gr])
            if P_full.has_colors() and colors_np_gr is not None:
                P_full.colors = o3d.utility.Vector3dVector(colors_np_gr[height_filter_mask_gr])
            if pcd_pixel_to_point_map.size > 0 and len(pcd_pixel_to_point_map) == len(height_filter_mask_gr): #确保长度一致
                 pcd_pixel_to_point_map = pcd_pixel_to_point_map[height_filter_mask_gr]
            else:
                 pcd_pixel_to_point_map = np.array([])
        if not P_full.has_points():
            if return_separate_segments:
                return (empty_pcd_dict_no_table, debug_images) if return_debug_images else empty_pcd_dict_no_table
            return (o3d.geometry.PointCloud(), debug_images) if return_debug_images else o3d.geometry.PointCloud()
    # print(f"{timing_prefix} 地面移除 耗时: {time.time() - t_start_ground_removal:.4f} 秒")

    num_total_points_full = len(P_full.points)
    final_selection_indices_mask = np.zeros(num_total_points_full, dtype=bool)

    t_start_hand_seg_full_part = time.time()
    hand_pcd_segment = o3d.geometry.PointCloud()
    best_mask_hand_2d = np.zeros(image_bgr_for_fastsam_model.shape[:2], dtype=bool)
    if text_prompt_hand and pcd_pixel_to_point_map.size > 0 :
        best_mask_hand_2d = get_fastsam_masks_from_text(text_prompt_hand, image_bgr_for_fastsam_model, fastsam_model_instance, effective_device, image_width_config_for_fastsam, conf_threshold=fastsam_conf_hand, parent_timing_prefix=timing_prefix)
        if np.any(best_mask_hand_2d):
            if return_debug_images:
                hand_mask_viz = image_bgr_for_fastsam_model.copy()
                hand_mask_viz[best_mask_hand_2d] = [0, 255, 0]
                debug_images['hand_mask_overlay'] = hand_mask_viz
            t_start_hand_mask_proc = time.time()
            mask_hand_resized_to_depth_dims = cv2.resize(
                best_mask_hand_2d.astype(np.uint8),
                (depth_image_input.shape[1], depth_image_input.shape[0]),
                interpolation=cv2.INTER_NEAREST).astype(bool)
            pixels_u_all = pcd_pixel_to_point_map[:, 0]; pixels_v_all = pcd_pixel_to_point_map[:, 1]
            valid_coords_mask = (pixels_u_all >= 0) & (pixels_u_all < mask_hand_resized_to_depth_dims.shape[1]) & \
                                  (pixels_v_all >= 0) & (pixels_v_all < mask_hand_resized_to_depth_dims.shape[0])
            valid_pcd_indices_for_masking = np.where(valid_coords_mask)[0]
            if valid_pcd_indices_for_masking.size > 0:
                coords_for_mask_lookup_u = pixels_u_all[valid_pcd_indices_for_masking]
                coords_for_mask_lookup_v = pixels_v_all[valid_pcd_indices_for_masking]
                sam_mask_values_at_pcd_points = mask_hand_resized_to_depth_dims[coords_for_mask_lookup_v, coords_for_mask_lookup_u]
                p_full_indices_for_hand = valid_pcd_indices_for_masking[sam_mask_values_at_pcd_points]
                if p_full_indices_for_hand.size > 0 and np.all(p_full_indices_for_hand < num_total_points_full):
                    final_selection_indices_mask[p_full_indices_for_hand.astype(int)] = True
                    hand_pcd_segment = P_full.select_by_index(p_full_indices_for_hand.astype(int))
            # print(f"{timing_prefix}    手部掩码后处理和点云选择 耗时: {time.time() - t_start_hand_mask_proc:.4f} 秒")
    # print(f"{timing_prefix} 手部整体分割部分 (含提示: '{text_prompt_hand}') 耗时: {time.time() - t_start_hand_seg_full_part:.4f} 秒") # 打印传入的提示

    t_start_obj_seg_full_part = time.time()
    object_pcd_segment = o3d.geometry.PointCloud()
    best_mask_object_2d = np.zeros(image_bgr_for_fastsam_model.shape[:2], dtype=bool)
    if text_prompt_object and pcd_pixel_to_point_map.size > 0:
        best_mask_object_2d = get_fastsam_masks_from_text(text_prompt_object, image_bgr_for_fastsam_model, fastsam_model_instance, effective_device, image_width_config_for_fastsam, conf_threshold=fastsam_conf_object, parent_timing_prefix=timing_prefix)
        if np.any(best_mask_object_2d):
            if return_debug_images:
                object_mask_viz = image_bgr_for_fastsam_model.copy()
                object_mask_viz[best_mask_object_2d] = [0, 0, 255]
                debug_images['object_mask_overlay'] = object_mask_viz
            t_start_obj_mask_proc = time.time()
            mask_object_resized_to_depth_dims = cv2.resize(
                best_mask_object_2d.astype(np.uint8),
                (depth_image_input.shape[1], depth_image_input.shape[0]),
                interpolation=cv2.INTER_NEAREST).astype(bool)
            pixels_u_all = pcd_pixel_to_point_map[:, 0]; pixels_v_all = pcd_pixel_to_point_map[:, 1]
            valid_coords_mask = (pixels_u_all >= 0) & (pixels_u_all < mask_object_resized_to_depth_dims.shape[1]) & \
                                  (pixels_v_all >= 0) & (pixels_v_all < mask_object_resized_to_depth_dims.shape[0])
            valid_pcd_indices_for_masking = np.where(valid_coords_mask)[0]
            if valid_pcd_indices_for_masking.size > 0:
                coords_for_mask_lookup_u = pixels_u_all[valid_pcd_indices_for_masking]
                coords_for_mask_lookup_v = pixels_v_all[valid_pcd_indices_for_masking]
                sam_mask_values_at_pcd_points = mask_object_resized_to_depth_dims[coords_for_mask_lookup_v, coords_for_mask_lookup_u]
                p_full_indices_for_object = valid_pcd_indices_for_masking[sam_mask_values_at_pcd_points]
                if p_full_indices_for_object.size > 0 and np.all(p_full_indices_for_object < num_total_points_full):
                    final_selection_indices_mask[p_full_indices_for_object.astype(int)] = True
                    object_pcd_segment = P_full.select_by_index(p_full_indices_for_object.astype(int))
            # print(f"{timing_prefix}    物体掩码后处理和点云选择 耗时: {time.time() - t_start_obj_mask_proc:.4f} 秒")
    # print(f"{timing_prefix} 物体整体分割部分 (含提示: '{text_prompt_object}') 耗时: {time.time() - t_start_obj_seg_full_part:.4f} 秒") # 打印传入的提示

    if return_debug_images and (np.any(best_mask_hand_2d) or np.any(best_mask_object_2d)):
        combined_mask_viz = image_bgr_for_fastsam_model.copy()
        if np.any(best_mask_hand_2d):
            combined_mask_viz[best_mask_hand_2d] = [0,255,0]
        if np.any(best_mask_object_2d):
            combined_mask_viz[best_mask_object_2d] = [0,0,255]
        debug_images['combined_masks_overlay'] = combined_mask_viz

    t_start_final_select_sor = time.time()
    final_selected_indices_from_mask = np.where(final_selection_indices_mask)[0]
    final_cropped_pcd = P_full.select_by_index(final_selected_indices_from_mask.astype(int))

    if final_cropped_pcd.has_points() and len(final_cropped_pcd.points) > 10:
        t_start_sor = time.time()
        cl, ind = final_cropped_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        final_cropped_pcd = final_cropped_pcd.select_by_index(ind)
        # print(f"{timing_prefix}    统计离群点移除 (SOR) 耗时: {time.time() - t_start_sor:.4f} 秒")

        if return_separate_segments:
            t_start_kdtree_filter = time.time()
            final_points_sor_np = np.asarray(final_cropped_pcd.points)

            num_hand_pts_before_kdtree = len(hand_pcd_segment.points) if hand_pcd_segment.has_points() else 0
            num_obj_pts_before_kdtree = len(object_pcd_segment.points) if object_pcd_segment.has_points() else 0
            # print(f"{timing_prefix}    KDTree同步前点数: 手部={num_hand_pts_before_kdtree}, 物体={num_obj_pts_before_kdtree}")

            if len(final_points_sor_np) > 0:
                pcd_tree_for_sor_filtered = o3d.geometry.KDTreeFlann(final_cropped_pcd)
                if hand_pcd_segment.has_points():
                    hand_points_original_np = np.asarray(hand_pcd_segment.points)
                    sor_hand_indices_in_original_segment = []
                    for point_idx in range(len(hand_points_original_np)):
                        point_to_check = hand_points_original_np[point_idx]
                        [k, idx_in_sor_cloud, _] = pcd_tree_for_sor_filtered.search_knn_vector_3d(point_to_check, 1)
                        if k > 0 and np.linalg.norm(point_to_check - final_points_sor_np[idx_in_sor_cloud[0]]) < 1e-6 :
                             sor_hand_indices_in_original_segment.append(point_idx)
                    if sor_hand_indices_in_original_segment:
                        hand_pcd_segment = hand_pcd_segment.select_by_index(sor_hand_indices_in_original_segment)
                    else:
                        hand_pcd_segment = o3d.geometry.PointCloud()
                else:
                    hand_pcd_segment = o3d.geometry.PointCloud()

                if object_pcd_segment.has_points():
                    object_points_original_np = np.asarray(object_pcd_segment.points)
                    sor_object_indices_in_original_segment = []
                    for point_idx in range(len(object_points_original_np)):
                        point_to_check = object_points_original_np[point_idx]
                        [k, idx_in_sor_cloud, _] = pcd_tree_for_sor_filtered.search_knn_vector_3d(point_to_check, 1)
                        if k > 0 and np.linalg.norm(point_to_check - final_points_sor_np[idx_in_sor_cloud[0]]) < 1e-6:
                            sor_object_indices_in_original_segment.append(point_idx)
                    if sor_object_indices_in_original_segment:
                        object_pcd_segment = object_pcd_segment.select_by_index(sor_object_indices_in_original_segment)
                    else:
                        object_pcd_segment = o3d.geometry.PointCloud()
                else:
                    object_pcd_segment = o3d.geometry.PointCloud()
            else:
                hand_pcd_segment = o3d.geometry.PointCloud()
                object_pcd_segment = o3d.geometry.PointCloud()
            # print(f"{timing_prefix}    KDTree同步SOR结果到手/物体点云段 耗时: {time.time() - t_start_kdtree_filter:.4f} 秒")

    # print(f"{timing_prefix} 最终点云选择和SOR (及可选的子点云段更新) 耗时: {time.time() - t_start_final_select_sor:.4f} 秒")

    pcd_results_dict = {
        'hand': hand_pcd_segment,
        'object': object_pcd_segment,
        'combined': final_cropped_pcd
    }

    # print(f"{timing_prefix} segment_and_crop_with_fastsam 函数总耗时: {time.time() - overall_func_start_time:.4f} 秒")

    if return_separate_segments:
        return (pcd_results_dict, debug_images) if return_debug_images else pcd_results_dict
    else:
        return (final_cropped_pcd, debug_images) if return_debug_images else final_cropped_pcd