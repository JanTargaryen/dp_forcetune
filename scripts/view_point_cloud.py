import zarr
import numpy as np
import os
# import open3d as o3d # 暂时不需要，除非要看非零点云
# import plotly.graph_objects as go # 暂时不需要
# import plotly.io as pio # 暂时不需要

# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
ZARR_DATASET_PATH = '/data1/zhoufang/dataset_output/processed_data_fastsam_cloud.zarr' # 您生成的Zarr文件路径
OUTPUT_POINTCLOUD_HTML_PATH = 'first_frame_pointcloud_from_zarr.html' # 修改一下文件名

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

    if 'data' not in root or 'point_cloud' not in root['data']:
        print("错误: Zarr数据集中缺少 'data/point_cloud'。")
        return
    
    point_cloud_dataset = root['data/point_cloud']
    
    if point_cloud_dataset.shape[0] == 0:
        print("错误: 'data/point_cloud' 数据集为空 (没有帧数据)。")
        return

    first_frame_global_idx = 0
    print(f"\n--- 查看Zarr文件中全局索引为 {first_frame_global_idx} 的点云数据 ---")

    point_cloud_data = point_cloud_dataset[first_frame_global_idx]
    print(f"点云数据形状: {point_cloud_data.shape}")
    print(f"点云数据类型: {point_cloud_data.dtype}")
    
    # 检查点云是否全为零
    if np.all(point_cloud_data == 0):
        print("警告: 从Zarr文件中读取的第一帧点云数据 *全部为零*！")
        print("这表明在数据转换过程中，点云未能正确生成或被零填充。")
        print("请检查 'convert_data_isaacgym.py' 脚本中的以下部分：")
        print("  1. 原始深度图数据是否有效 (是否全为零或无效值)。")
        print("  2. K_MATRIX (相机内参) 是否准确。")
        print("  3. CAMERA_WORLD_POSE (相机外参) 是否准确。")
        print("  4. Open3D从深度图创建点云的参数 (depth_scale, depth_trunc) 是否合适。")
        print("  5. 点云从相机坐标系 -> 世界坐标系 -> 机器人基座坐标系的转换是否正确。")
        print("  6. 点云采样/填充逻辑是否在原始点云为空时导致全零填充。")
    else:
        print("从Zarr文件中读取的第一帧点云数据包含非零值。")
        print("前5个点:")
        print(point_cloud_data[:5, :])
        
        # (可选) 如果点云非零，再尝试保存HTML进行可视化
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            trace = go.Scatter3d(
                x=point_cloud_data[:, 0], y=point_cloud_data[:, 1], z=point_cloud_data[:, 2],
                mode='markers', marker=dict(size=1)
            )
            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis=dict(title='X (base frame)'), yaxis=dict(title='Y (base frame)'),
                    zaxis=dict(title='Z (base frame)'), aspectmode='data'
                )
            )
            fig = go.Figure(data=[trace], layout=layout)
            pio.write_html(fig, file=OUTPUT_POINTCLOUD_HTML_PATH, auto_open=False)
            print(f"点云可视化已保存到: {os.path.abspath(OUTPUT_POINTCLOUD_HTML_PATH)}")
        except ImportError:
            print("警告: `plotly` 未安装。无法保存点云为HTML。")
        except Exception as e:
            print(f"错误: 保存点云为HTML时出错: {e}")

    # 您也可以在这里加上打印状态和动作的代码，以确认它们是否也被正确读取
    if 'state' in root['data'] and root['data/state'].shape[0] > 0:
        state_data = root['data/state'][first_frame_global_idx]
        print(f"\n状态数据 (形状: {state_data.shape}):\n{state_data}")
    if 'action' in root['data'] and root['data/action'].shape[0] > 0:
        action_data = root['data/action'][first_frame_global_idx]
        print(f"\n动作数据 (形状: {action_data.shape}):\n{action_data}")


if __name__ == '__main__':
    main()