import zarr
import numpy as np

def inspect_zarr_dataset(zarr_path):
    """
    检查Zarr数据集的结构和潜在的数据冗余。

    Args:
        zarr_path (str): Zarr文件的路径。
    """
    print(f"开始检查Zarr文件: {zarr_path}\n")

    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"错误: 无法打开Zarr文件 '{zarr_path}'. 异常: {e}")
        return

    # 1. 检查基本结构
    print("Zarr文件顶层结构:", list(root.keys()))
    if 'data' not in root or 'meta' not in root:
        print("错误: Zarr文件缺少 'data' 或 'meta' 组。")
        return

    data_group = root['data']
    meta_group = root['meta']
    print(" 'data' 组包含:", list(data_group.keys()))
    print(" 'meta' 组包含:", list(meta_group.keys()))

    required_data_keys = ['point_cloud', 'state', 'action']
    for key in required_data_keys:
        if key not in data_group:
            print(f"警告: 'data' 组中缺少必要的数据集 '{key}'。")
            # 可以选择在这里返回，或者继续检查存在的数据集

    if 'episode_ends' not in meta_group:
        print("错误: 'meta' 组中缺少 'episode_ends' 数据集。无法进行逐episode检查。")
        # 即使没有 episode_ends，也可以进行总帧数检查
    else:
        episode_ends = meta_group['episode_ends'][:]
        print(f"\n 'meta/episode_ends' 数据 (前10条): {episode_ends[:10]}")
        print(f" 'meta/episode_ends' 总条目数 (即总episode数): {len(episode_ends)}")

    # 2. 打印并比较主要数据集的总形状和总帧数
    print("\n--- 数据集总帧数检查 ---")
    total_frames = {}
    datasets_to_check = {}

    if 'point_cloud' in data_group:
        datasets_to_check['point_cloud'] = data_group['point_cloud']
    if 'state' in data_group:
        datasets_to_check['state'] = data_group['state']
    if 'action' in data_group:
        datasets_to_check['action'] = data_group['action']

    if not datasets_to_check:
        print("错误: 'data' 组中没有可供检查的数据集 ('point_cloud', 'state', 'action')。")
        return

    for name, dset in datasets_to_check.items():
        print(f"数据集 '{name}':")
        print(f"  - 形状: {dset.shape}")
        print(f"  - 总帧数: {dset.shape[0]}")
        total_frames[name] = dset.shape[0]

    # 比较总帧数
    unique_frame_counts = set(total_frames.values())
    if len(unique_frame_counts) > 1:
        print("\n警告: 主要数据集的总帧数不一致！这可能表明数据损坏或处理错误。")
        for name, count in total_frames.items():
            print(f"  - '{name}' 总帧数: {count}")
    elif unique_frame_counts:
        print(f"\n信息: 主要数据集的总帧数一致，均为: {list(unique_frame_counts)[0]}")
    else:
        print("\n信息: 没有可比较总帧数的数据集。")


    # 3. 逐个episode检查帧数 (如果 episode_ends 存在)
    if 'episode_ends' in meta_group and datasets_to_check:
        print("\n--- 逐Episode帧数检查 (与 episode_ends 比较) ---")
        num_episodes = len(episode_ends)
        if num_episodes == 0:
            print("信息: 'episode_ends' 为空，没有episode可供检查。")
            return

        start_idx = 0
        any_episode_discrepancy_found = False

        for i in range(num_episodes):
            end_idx = episode_ends[i]
            # episode_ends 存储的是包含当前帧的索引，所以长度是 end_idx - start_idx + 1
            expected_ep_len = end_idx - start_idx + 1
            # print(f"\nEpisode {i+1}:")
            # print(f"  - 'episode_ends' 记录的结束索引: {end_idx}")
            # print(f"  - 预期帧范围 (0-based): 从 {start_idx} 到 {end_idx}")
            # print(f"  - 'episode_ends' 计算得到的预期长度: {expected_ep_len}")

            episode_frame_counts = {}
            discrepancy_in_this_episode = False

            for name, dset in datasets_to_check.items():
                # 检查索引是否越界
                if start_idx >= dset.shape[0] or end_idx >= dset.shape[0]:
                    # print(f"  警告: Episode {i+1} 的索引范围 [{start_idx}-{end_idx}] 超出了数据集 '{name}' (总帧数 {dset.shape[0]}) 的界限。")
                    actual_ep_len = -1 # 表示无法获取
                    discrepancy_in_this_episode = True # 标记问题
                else:
                    # 实际从数据集中切片得到的该episode的长度
                    # 注意：Zarr切片是包含起始，不包含结束，所以是 end_idx + 1
                    # 但由于我们是基于 episode_ends (它是包含的索引)，所以长度是 end_idx - start_idx + 1
                    # 假设 dset[start_idx : end_idx+1] 是正确的切片方式
                    actual_ep_len = dset[start_idx : end_idx + 1].shape[0]
                
                episode_frame_counts[name] = actual_ep_len

                if actual_ep_len != -1 and actual_ep_len != expected_ep_len:
                    discrepancy_in_this_episode = True
            
            if discrepancy_in_this_episode:
                any_episode_discrepancy_found = True
                print(f"\n警告: Episode {i+1} 存在帧数不匹配:")
                print(f"  - 'episode_ends' 计算得到的预期长度: {expected_ep_len} (帧索引从 {start_idx} 到 {end_idx})")
                for name, count in episode_frame_counts.items():
                    if count == -1:
                        print(f"    - '{name}' 数据: 无法获取 (索引越界)")
                    else:
                        print(f"    - '{name}' 数据实际长度: {count}")
                        if count > expected_ep_len:
                            print(f"      -> !!! '{name}' 的数据可能存在冗余帧 ({count - expected_ep_len} 帧多余) !!!")
                        elif count < expected_ep_len:
                            print(f"      -> ??? '{name}' 的数据可能存在缺失帧 ({expected_ep_len - count} 帧缺失) ???")
            
            # 更新下一轮的起始索引
            start_idx = end_idx + 1
        
        if not any_episode_discrepancy_found:
            print("\n信息: 所有Episode的帧数与 'episode_ends' 记录一致。未发现明显的冗余或缺失。")

    elif not datasets_to_check:
        print("\n信息: 由于缺少主要数据集，无法进行逐episode检查。")
    else: # episode_ends 不存在
        print("\n信息: 'meta/episode_ends' 不存在，跳过逐episode帧数检查。")


    # 4. 检查 state 和 action 是否可能比 point_cloud 多帧 (直接比较总数)
    #    这种情况可能发生在 episode_ends 记录正确，但 state/action 在最后一个 episode 后还有多余数据
    if 'point_cloud' in total_frames and 'state' in total_frames:
        if total_frames['state'] > total_frames['point_cloud']:
            print(f"\n严重警告: 'state' 数据集总帧数 ({total_frames['state']}) "
                  f"大于 'point_cloud' 数据集总帧数 ({total_frames['point_cloud']})!")
            print("  这强烈表明 'state' 数据存在冗余。")
    if 'point_cloud' in total_frames and 'action' in total_frames:
        if total_frames['action'] > total_frames['point_cloud']:
            print(f"\n严重警告: 'action' 数据集总帧数 ({total_frames['action']}) "
                  f"大于 'point_cloud' 数据集总帧数 ({total_frames['point_cloud']})!")
            print("  这强烈表明 'action' 数据存在冗余。")


    print("\n--- 检查完毕 ---")

if __name__ == '__main__':
    # 请将下面的路径替换为您实际的 processed_data.zarr 文件路径
    zarr_file_path = '/data1/zhoufang/dataset_output/processed_data.zarr'
    # 例如: zarr_file_path = 'path/to/your/processed_data.zarr'
    
    inspect_zarr_dataset(zarr_file_path)