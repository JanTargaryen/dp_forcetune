#!/bin/bash

# ==========================================================================================
# 自定义 DP3 任务训练脚本 (run_xarm_training.sh)
#
# 说明:
# 这个脚本用于启动您的自定义 xArm Inspire grasp 任务的 DP3 训练。
# 它基于您提供的 train_policy.sh 示例进行修改。
#
# 使用方法:
# bash scripts/run_xarm_training.sh <算法名称> <任务配置名> <附加实验信息> <随机种子> <GPU ID>
#
# 参数:
#   <算法名称>:         您想使用的算法/主配置文件名 (例如: "dp3" 或 "simple_dp3")。
#                        脚本会加载 "configs/<算法名称>.yaml" 或 "diffusion_policy_3d/config/<算法名称>.yaml"。
#   <任务配置名>:       您的任务配置文件名 (不含.yaml后缀，例如: "xarm_inspire_grasp_task")。
#                        脚本会通过 Hydra 的 task= 参数覆盖默认任务。
#   <附加实验信息>:   用于构成实验名称的附加字符串 (例如: "v1_longer_horizon")。
#   <随机种子>:         训练使用的随机种子 (例如: 42)。
#   <GPU ID>:           要使用的GPU的ID (例如: 0)。
#
# 示例:
# bash scripts/run_xarm_training.sh dp3 xarm_inspire_grasp_task initial_run 42 0
# ==========================================================================================

# --- 默认设置 ---
DEBUG_MODE=False # 设置为True以启用调试模式 (例如，wandb offline)
SAVE_CHECKPOINTS=True # 是否保存模型检查点

# --- 参数检查 ---
if [ "$#" -ne 5 ]; then
    echo "错误: 需要 5 个参数。"
    echo "用法: bash scripts/run_xarm_training.sh <算法名称> <任务配置名> <附加实验信息> <随机种子> <GPU ID>"
    echo "示例: bash scripts/run_xarm_training.sh dp3 xarm_inspire_grasp_task initial_run 42 0"
    exit 1
fi

# --- 从命令行参数赋值 ---
ALG_NAME=$1             # 例如: dp3
TASK_CONFIG_NAME=$2     # 例如: xarm_inspire_grasp_task (对应 configs/task/xarm_inspire_grasp_task.yaml)
ADDITIONAL_INFO=$3      # 例如: initial_run
SEED=$4                 # 例如: 42
GPU_ID=$5               # 例如: 0

# --- 构建配置和路径 ---
# CONFIG_NAME_FOR_HYDRA 通常是主配置文件，例如 dp3.yaml
# train.py 的 --config-name 参数期望的是不含 .yaml 的文件名
CONFIG_NAME_FOR_HYDRA="${ALG_NAME}" # Hydra 会自动寻找 configs/${ALG_NAME}.yaml 或类似路径

# 实验名称和运行目录
# 注意: ${task_name} 在Hydra命令中会被解析为实际任务名(如果主配置中有定义并被覆盖)
# 这里我们用传入的 TASK_CONFIG_NAME 来构建一个近似的目录结构，
# 最终的目录结构由Hydra的 hydra.run.dir 或 hydra.sweep.dir 控制。
EXPERIMENT_NAME="${TASK_CONFIG_NAME}-${ALG_NAME}-${ADDITIONAL_INFO}"
RUN_DIR_BASE="data/outputs" # 与官方dp3.yaml中的hydra.run.dir基础路径一致
RUN_DIR="${RUN_DIR_BASE}/${EXPERIMENT_NAME}_seed${SEED}" # 训练输出将保存在此目录下

# --- 设置W&B模式 ---
if [ $DEBUG_MODE = True ]; then
    WANDB_MODE="offline"
    echo -e "\033[33m启用调试模式! Weights & Biases 将在离线模式下运行。\033[0m"
else
    WANDB_MODE="online"
    echo -e "\033[32m启用训练模式。Weights & Biases 将在在线模式下运行。\033[0m"
fi

# --- 打印将要执行的配置 ---
echo ""
echo -e "\033[1;34m=================================================="
echo "           开始训练 DP3 模型"
echo "--------------------------------------------------"
echo "主配置 (算法): ${CONFIG_NAME_FOR_HYDRA}"
echo "任务配置覆盖:  ${TASK_CONFIG_NAME}"
echo "实验名称:      ${EXPERIMENT_NAME}"
echo "随机种子:      ${SEED}"
echo "GPU ID (使用): ${GPU_ID}"
echo "运行目录:      ${RUN_DIR}"
echo "W&B 模式:      ${WANDB_MODE}"
echo "保存检查点:    ${SAVE_CHECKPOINTS}"
echo "==================================================\033[0m"
echo ""

# --- 设置环境变量 ---
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# --- 执行训练命令 ---
# 确保 train.py 在当前目录下，或者 python 命令能找到它。
# --config-name 通常指向一个主配置文件 (例如 dp3.yaml)，它内部定义了默认的 task。
# task=${TASK_CONFIG_NAME} 会覆盖主配置文件中默认的 task。
# (在之前的交互中，您确认了您的任务配置文件名为 xarm_inspire_grasp_task.yaml)

echo "正在执行: python train.py --config-name=${CONFIG_NAME_FOR_HYDRA} task=${TASK_CONFIG_NAME} hydra.run.dir=${RUN_DIR} training.debug=${DEBUG_MODE} training.seed=${SEED} training.device=\"cuda:0\" exp_name=${EXPERIMENT_NAME} logging.mode=${WANDB_MODE} checkpoint.save_ckpt=${SAVE_CHECKPOINTS}"
echo ""

python train.py --config-name=${CONFIG_NAME_FOR_HYDRA} \
                task=${TASK_CONFIG_NAME} \
                hydra.run.dir=${RUN_DIR} \
                training.debug=${DEBUG_MODE} \
                training.seed=${SEED} \
                training.device="cuda:0" \
                exp_name=${EXPERIMENT_NAME} \
                logging.mode=${WANDB_MODE} \
                checkpoint.save_ckpt=${SAVE_CHECKPOINTS}

EXIT_CODE=$?

echo ""
echo -e "\033[1;34m=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\033[32m训练脚本执行完毕。\033[0m"
else
    echo -e "\033[31m训练脚本执行出错! 退出码: ${EXIT_CODE}\033[0m"
fi
echo "请检查输出目录: ${RUN_DIR}"
echo "以及Weights & Biases (wandb.ai) 上的日志 (项目名应为 DP3_xarm_inspire_grasp 或类似)。"
echo "==================================================\033[0m"

exit $EXIT_CODE
