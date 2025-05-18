import zarr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # 用于处理图像数组和保存为PNG
import open3d as o3d
import os
import json # 用于查找原始JSON以获取图像路径

# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
ZARR_DATASET_PATH = '/data1/zhoufang/dataset_output/processed_grasp_data.zarr' # 您生成的Zarr文件路径
# 原始数据目录，用于查找图像.npy文件 (需要与convert_data_isaacgym.py中的DATA_DIR一致)
ORIGINAL_DATA_DIR = '/data1/zhoufang/dataset/collected_data_teleop'
# 如果原始的trajectory_*.json文件也需要访问以获取精确的图像路径，可以设置
# ORIGINAL_TRAJECTORY_FILES_PATTERN = os.path.join(ORIGINAL_DATA_DIR, '**', 'trajectory_*.json') # 使用递归查找

OUTPUT_IMAGE_PATH = 'first_frame_rgb.png'
OUTPUT_POINTCLOUD_HTML_PATH = 'first_frame_pointcloud.html'

# ---------------------------------------------------------------------------
# 2. 辅助函数 (可选，如果需要从原始JSON查找图像路径)
# ---------------------------------------------------------------------------
def find_original_frame_data(target_frame_idx, original_data_dir):
    """
    尝试根据全局帧索引找到对应的原始JSON文件和帧内索引，以获取图像路径。
    注意：这是一个简化的查找，假设Zarr文件中的帧顺序与原始轨迹文件顺序一致。
    """
    all_traj_files = []
    abs_original_data_dir = os.path.abspath(original_data_dir)
    for root_dir, _, files in os.walk(abs_original_data_dir):
        for filename in files:
            if filename.startswith('trajectory_') and filename.endswith('.json'):
                all_traj_files.append(os.path.join(root_dir, filename))
    all_traj_files = sorted(all_traj_files)

    current_frame_count = 0
    for traj_file_path in all_traj_files:
        try:
            with open(traj_file_path, 'r') as f:
                trajectory_data = json.load(f)
            if not isinstance(trajectory_data, list):
                continue
            
            num_frames_in_this_traj = 0
            # 计算该轨迹中实际被转换脚本处理的有效帧数
            # (这部分比较复杂，因为转换脚本可能会跳过无效帧)
            # 为了简化，我们先假设所有帧都有效，或者需要更复杂的映射
            # 一个更简单（但不完全准确，除非所有帧都有效）的方法是：
            # 如果 target_frame_idx < current_frame_count + len(trajectory_data):
            #    frame_in_traj_idx = target_frame_idx - current_frame_count
            #    return trajectory_data[frame_in_traj_idx], os.path.dirname(traj_file_path)
            # current_frame_count += len(trajectory_data)

            # 更准确的方法是，我们需要知道转换脚本是如何跳过帧的，或者
            # DataCollector在保存时就保证了rgb_image_path的有效性，并且这个路径
            # 是相对于 DATA_DIR 或者轨迹文件本身的。

            # 鉴于您JSON中rgb_image_path是 "images/rgb_XXXX.npy"，
            # 它很可能是相对于该JSON文件所在的目录。
            if target_frame_idx < current_frame_count + len(trajectory_data):
                frame_in_traj_idx = target_frame_idx - current_frame_count
                if 0 <= frame_in_traj_idx < len(trajectory_data):
                    return trajectory_data[frame_in_traj_idx], os.path.dirname(traj_file_path)
            current_frame_count += len(trajectory_data)


        except Exception:
            continue # 跳过无法处理的文件
    return None, None


# ---------------------------------------------------------------------------
# 3. 主逻辑
# ---------------------------------------------------------------------------
def main():
    # 检查Zarr文件是否存在
    if not os.path.exists(ZARR_DATASET_PATH):
        print(f"错误: Zarr数据集 '{ZARR_DATASET_PATH}' 未找到。请先运行数据转换脚本。")
        return

    # 打开Zarr数据集
    try:
        root = zarr.open_group(ZARR_DATASET_PATH, mode='r')
    except Exception as e:
        print(f"错误: 无法打开Zarr数据集 '{ZARR_DATASET_PATH}'。错误: {e}")
        return

    print("成功打开Zarr数据集。")
    print("数据集结构:")
    print(root.tree())

    # 检查必要的数据组是否存在
    if 'data' not in root:
        print("错误: Zarr数据集中缺少 'data' 组。")
        return
    
    data_group = root['data']
    required_data_keys = ['point_cloud', 'state', 'action']
    for key in required_data_keys:
        if key not in data_group:
            print(f"错误: 'data' 组中缺少 '{key}' 数据集。")
            return

    # 读取第一条轨迹的第一帧数据
    # 首先，我们需要知道第一条轨迹在哪里结束
    if 'meta' not in root or 'episode_ends' not in root['meta']:
        print("错误: Zarr数据集中缺少 'meta/episode_ends'。无法确定第一条轨迹。")
        # 我们可以尝试读取全局索引为0的帧，但这不一定是第一条轨迹的第一帧
        # 如果没有episode_ends，我们就假设第一条轨迹至少包含第一帧
        first_frame_global_idx = 0
        print("警告: 缺少 episode_ends，将尝试读取全局索引为0的帧作为第一帧。")
    else:
        episode_ends = root['meta/episode_ends'][:]
        if len(episode_ends) == 0:
            print("错误: 'episode_ends' 为空，无法确定轨迹。")
            # 同上，尝试读取全局索引0
            first_frame_global_idx = 0
            print("警告: episode_ends 为空，将尝试读取全局索引为0的帧作为第一帧。")

        else:
            # 第一条轨迹的第一帧就是全局索引0
            first_frame_global_idx = 0 
            # first_episode_end_idx = episode_ends[0]
            # print(f"第一条轨迹从索引 0 到 {first_episode_end_idx}。")


    print(f"\n--- 查看全局索引为 {first_frame_global_idx} 的数据 (通常是第一条轨迹的第一帧) ---")

    # 1. 读取并打印点云数据
    if 'point_cloud' in data_group:
        if first_frame_global_idx < data_group['point_cloud'].shape[0]:
            point_cloud_data = data_group['point_cloud'][first_frame_global_idx]
            print(f"\n点云数据 (形状: {point_cloud_data.shape}, 前5个点):")
            print(point_cloud_data[:5, :])

            # 保存点云为HTML
            if point_cloud_data.size > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
                # 可以添加颜色，例如所有点为蓝色
                # pcd.paint_uniform_color([0, 0, 1]) 
                
                # 使用Open3D内置方法保存交互式HTML (如果可用且简单)
                # o3d.visualization.draw_geometries([pcd]) # 这会打开一个窗口
                # 保存HTML通常需要更复杂的plotly集成，如3D-Diffusion-Policy/data/visualize.py所示
                # 这里我们使用一个简单的方式来生成HTML，或者提示用户使用现有工具
                try:
                    # 尝试模仿 3D-Diffusion-Policy/data/visualize.py 中的保存方式
                    # 这需要 plotly
                    import plotly.graph_objects as go
                    import plotly.io as pio

                    trace = go.Scatter3d(
                        x=point_cloud_data[:, 0],
                        y=point_cloud_data[:, 1],
                        z=point_cloud_data[:, 2],
                        mode='markers',
                        marker=dict(size=1) # 调整点的大小
                    )
                    layout = go.Layout(
                        margin=dict(l=0, r=0, b=0, t=0),
                        scene=dict(
                            xaxis=dict(title='X (base frame)'),
                            yaxis=dict(title='Y (base frame)'),
                            zaxis=dict(title='Z (base frame)'),
                            aspectmode='data' #保持轴的比例
                        )
                    )
                    fig = go.Figure(data=[trace], layout=layout)
                    pio.write_html(fig, file=OUTPUT_POINTCLOUD_HTML_PATH, auto_open=False)
                    print(f"点云可视化已保存到: {os.path.abspath(OUTPUT_POINTCLOUD_HTML_PATH)}")

                except ImportError:
                    print("警告: `plotly` 未安装。无法保存点云为HTML。请运行 `pip install plotly`。")
                except Exception as e:
                    print(f"错误: 保存点云为HTML时出错: {e}")
            else:
                print("点云数据为空，不保存HTML。")
        else:
            print(f"错误: 索引 {first_frame_global_idx} 超出点云数据范围。")
    else:
        print("数据集中没有 'point_cloud'。")


    # 2. 读取并打印状态数据
    if 'state' in data_group:
        if first_frame_global_idx < data_group['state'].shape[0]:
            state_data = data_group['state'][first_frame_global_idx]
            print(f"\n状态数据 (形状: {state_data.shape}):")
            print(state_data)
        else:
            print(f"错误: 索引 {first_frame_global_idx} 超出状态数据范围。")
    else:
        print("数据集中没有 'state'。")


    # 3. 读取并打印动作数据
    if 'action' in data_group:
        if first_frame_global_idx < data_group['action'].shape[0]:
            action_data = data_group['action'][first_frame_global_idx]
            print(f"\n动作数据 (形状: {action_data.shape}):")
            print(action_data)
        else:
            print(f"错误: 索引 {first_frame_global_idx} 超出动作数据范围。")
    else:
        print("数据集中没有 'action'。")


    # 4. 查找并加载RGB图像 (从.npy文件)
    # 我们需要找到原始JSON中对应这一帧的rgb_image_path
    original_frame_info, traj_dir_path = find_original_frame_data(first_frame_global_idx, ORIGINAL_DATA_DIR)

    if original_frame_info and 'rgb_image_path' in original_frame_info:
        rgb_relative_path = original_frame_info['rgb_image_path']
        # 假设rgb_image_path是相对于该JSON文件所在的目录(traj_dir_path)
        rgb_full_path = os.path.join(traj_dir_path, rgb_relative_path)
        
        if os.path.exists(rgb_full_path):
            try:
                rgb_array = np.load(rgb_full_path)
                print(f"\nRGB图像数据 (从 {rgb_full_path} 加载, 形状: {rgb_array.shape}, 类型: {rgb_array.dtype})")
                
                # 保存为PNG
                if rgb_array.ndim == 3 and rgb_array.shape[2] in [3, 4]: # H, W, C
                    # 如果是RGBA，取RGB部分
                    if rgb_array.shape[2] == 4:
                        rgb_array = rgb_array[:, :, :3]
                    
                    # 确保数据类型是 uint8 for Pillow
                    if rgb_array.dtype != np.uint8:
                        if np.issubdtype(rgb_array.dtype, np.floating) and rgb_array.max() <= 1.0 and rgb_array.min() >=0.0 :
                            rgb_array = (rgb_array * 255).astype(np.uint8)
                        else:
                            # 尝试进行基本的数据范围调整
                            rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
                            print("警告: RGB图像数据类型不是uint8，已尝试转换。请检查图像是否正确。")
                    
                    img = Image.fromarray(rgb_array)
                    img.save(OUTPUT_IMAGE_PATH)
                    print(f"RGB图像已保存到: {os.path.abspath(OUTPUT_IMAGE_PATH)}")
                else:
                    print(f"错误: RGB图像数组的形状 {rgb_array.shape} 不是预期的 HxWxC (C=3或4)。无法保存为PNG。")

            except FileNotFoundError:
                print(f"错误: RGB图像 .npy 文件未找到: {rgb_full_path}")
            except Exception as e:
                print(f"错误: 加载或保存RGB图像时出错: {e}")
        else:
            print(f"错误: 计算得到的RGB图像路径不存在: {rgb_full_path}")
    else:
        print(f"\n未能找到全局索引 {first_frame_global_idx} 对应的原始RGB图像路径信息，或路径无效。")


if __name__ == '__main__':
    main()