import zarr
import numpy as np
import os
# import open3d as o3d # 暂时不需要，除非要看非零点云
# import plotly.graph_objects as go # 动态导入
# import plotly.io as pio # 动态导入

NUM_VIEW_DATASET = 10 # 查看前多少条数据
# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
ZARR_DATASET_PATH = '/data1/zhoufang/dataset_output/processed_data.zarr' # 您生成的Zarr文件路径
OUTPUT_HTML_DIR = 'first_trajectory_pointclouds' # 保存HTML文件的输出目录

# ---------------------------------------------------------------------------
# 2. 主逻辑
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(ZARR_DATASET_PATH):
        print(f"错误: Zarr数据集 '{ZARR_DATASET_PATH}' 未找到。")
        return

    try:
        root = zarr.open_group(ZARR_DATASET_PATH, mode='r')
    except Exception as e:
        print(f"错误: 无法打开Zarr数据集 '{ZARR_DATASET_PATH}'。错误: {e}")
        return

    print("成功打开Zarr数据集。")
    print("数据集结构:")
    print(root.tree())

    # 检查必要的组和数据集是否存在
    if 'data' not in root or 'point_cloud' not in root['data']:
        print("错误: Zarr数据集中缺少 'data/point_cloud'。")
        return
    if 'meta' not in root or 'episode_ends' not in root['meta']:
        print("错误: Zarr数据集中缺少 'meta/episode_ends'，无法确定轨迹边界。")
        return
    
    point_cloud_dataset = root['data/point_cloud']
    episode_ends_dataset = root['meta/episode_ends']
    
    if point_cloud_dataset.shape[0] == 0:
        print("错误: 'data/point_cloud' 数据集为空 (没有帧数据)。")
        return
    if episode_ends_dataset.shape[0] == 0:
        print("错误: 'meta/episode_ends' 数据集为空，无法确定轨迹。")
        return

    # 创建输出目录 (如果不存在)
    if not os.path.exists(OUTPUT_HTML_DIR):
        try:
            os.makedirs(OUTPUT_HTML_DIR)
            print(f"创建输出目录: {OUTPUT_HTML_DIR}")
        except Exception as e:
            print(f"错误: 无法创建输出目录 '{OUTPUT_HTML_DIR}'. 错误: {e}")
            return

    first_trajectory_end_global_idx = episode_ends_dataset[0]
    num_frames_in_first_trajectory = first_trajectory_end_global_idx + 1
    num_frames_to_process = min(NUM_VIEW_DATASET, num_frames_in_first_trajectory)

    print(f"\n--- 查看第一条轨迹的前 {num_frames_to_process} 帧数据 ---")
    print(f"第一条轨迹包含 {num_frames_in_first_trajectory} 帧 (全局索引从 0 到 {first_trajectory_end_global_idx}).")

    for frame_local_idx in range(num_frames_to_process):
        current_global_frame_idx = frame_local_idx 

        print(f"\n--- 轨迹 0, 帧 {frame_local_idx} (全局索引 {current_global_frame_idx}) ---")

        if current_global_frame_idx >= point_cloud_dataset.shape[0]:
            print(f"警告: 请求的全局索引 {current_global_frame_idx} 超出点云数据集范围 ({point_cloud_dataset.shape[0]} 帧)。")
            continue

        point_cloud_data = point_cloud_dataset[current_global_frame_idx]
        print(f"点云数据形状: {point_cloud_data.shape}")
        print(f"点云数据类型: {point_cloud_data.dtype}")
        
        is_all_zero_check = np.all(np.abs(point_cloud_data) < 1e-9) # 使用容差检查是否几乎全为零

        if is_all_zero_check: # 使用带容差的检查
            print("警告: 当前帧的点云数据 *几乎全部为零*！")
        else:
            print("当前帧的点云数据包含非零值。")
            print("前2个点 (XYZ):") 
            print(point_cloud_data[:2, :3]) 
            if point_cloud_data.shape[1] >= 6:
                print("前2个点的颜色 (RGB):")
                print(point_cloud_data[:2, 3:6])
            
            output_html_filename = f"trajectory_0_frame_{frame_local_idx:02d}.html"
            output_html_filepath = os.path.join(OUTPUT_HTML_DIR, output_html_filename)
            
            try:
                import plotly.graph_objects as go
                import plotly.io as pio

                x_coords = point_cloud_data[:, 0]
                y_coords = point_cloud_data[:, 1]
                z_coords = point_cloud_data[:, 2]
                
                marker_config = dict(size=1)
                
                # --- 修改开始：应用颜色 ---
                if point_cloud_data.shape[1] >= 6: # 确保有颜色数据列
                    colors_rgb = point_cloud_data[:, 3:6] # 提取 R, G, B 列
                    
                    # 检查颜色值范围，如果不是 [0,1]，可能需要调整
                    # 例如，如果值是 [0,255]，则需要 colors_rgb = colors_rgb / 255.0
                    # 但通常来自Open3D等库的点云颜色已经是 [0,1] 的浮点数
                    
                    # 确保颜色值在有效范围内 (例如，裁剪到0-1，防止Plotly出错)
                    colors_rgb = np.clip(colors_rgb, 0.0, 1.0)
                    
                    marker_config['color'] = colors_rgb # 将Nx3的RGB颜色数组直接赋给marker的color属性
                else:
                    print("    点云数据不包含颜色信息 (列数 < 6)。将使用默认颜色。")
                # --- 修改结束 ---

                trace = go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='markers', marker=marker_config
                )
                layout = go.Layout(
                    title=f"Trajectory 0, Frame {frame_local_idx} (Global {current_global_frame_idx})",
                    margin=dict(l=0, r=0, b=0, t=0),
                    scene=dict(
                        xaxis=dict(title='X'), yaxis=dict(title='Y'),
                        zaxis=dict(title='Z'), aspectmode='data'
                    )
                )
                fig = go.Figure(data=[trace], layout=layout)
                pio.write_html(fig, file=output_html_filepath, auto_open=False)
                print(f"点云可视化已保存到: {os.path.abspath(output_html_filepath)}")
            except ImportError:
                print("警告: `plotly` 未安装。无法保存点云为HTML。")
            except Exception as e:
                print(f"错误: 保存点云为HTML时出错 (文件: {output_html_filepath}): {e}")

        # 打印状态和动作数据
        if 'state' in root['data'] and current_global_frame_idx < root['data/state'].shape[0]:
            state_data = root['data/state'][current_global_frame_idx]
            print(f"状态数据 (形状: {state_data.shape}): {state_data[:min(5, len(state_data))]}{'...' if len(state_data) > 5 else ''}")
        else:
            print("未找到或索引超出范围: 状态数据")

        if 'action' in root['data'] and current_global_frame_idx < root['data/action'].shape[0]:
            action_data = root['data/action'][current_global_frame_idx]
            print(f"动作数据 (形状: {action_data.shape}): {action_data[:min(5, len(action_data))]}{'...' if len(action_data) > 5 else ''}")
        else:
            print("未找到或索引超出范围: 动作数据")

if __name__ == '__main__':
    main()
