#!/bin/bash

# ########################################
# # Configurable via command-line args
# ########################################

# Default values
DATA_PATH="/mnt/longcontext/models/siyuan/rl_datasets/longcontext_train_30k/train.jsonl"
TOKENIZER_PATH="/mnt/longcontext/models/siyuan/llama3/llama-3.1-8B-instruct"
MAX_SAMPLES=10000
N_SAMPLES=8
EPISODE=1
WARMUP=0.0
TBS=512
RBS=128
KL=0.001
LR=2e-6
MAX_LENGTH=4096
PROMPT_MAX_LENGTH=8192
PORT=1278
TEMP=0.6
SAVE_MODEL_NAME_PREFIX="trainall-llama31-8b"
REWARD_MODEL="server_llama3_reward"
GROUP_METHOD="normal"
LOG_BASE="log"
NODE_RANK=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --tokenizer_path)
      TOKENIZER_PATH="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --n_samples)
      N_SAMPLES="$2"
      shift 2
      ;;
    --episode)
      EPISODE="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --tbs)
      TBS="$2"
      shift 2
      ;;
    --rbs)
      RBS="$2"
      shift 2
      ;;
    --kl)
      KL="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --max_length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --prompt_max_length)
      PROMPT_MAX_LENGTH="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --temp)
      TEMP="$2"
      shift 2
      ;;
    --save_model_name_prefix)
      SAVE_MODEL_NAME_PREFIX="$2"
      shift 2
      ;;
    --reward_model)
      REWARD_MODEL="$2"
      shift 2
      ;;
    --group_method)
      GROUP_METHOD="$2"
      shift 2
      ;;
    --log_base)
      LOG_BASE="$2"
      shift 2
      ;;
    --node_rank)
      NODE_RANK="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Your wandb token (presumed to be already set)
wandb_token=$WANDB_TOKEN
sudo rm -rf ~/.netrc

pkill -f -9 openrlhf

# Create necessary directories
mkdir -p results/$SAVE_MODEL_NAME
mkdir -p results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/

pkill -f ray

if [ "$NODE_RANK" = "0" ]; then
    ray start --head --port=8265 --dashboard-port=8266 --object-manager-port=8280 --node-manager-port=8290 --num-cpus=64 --num-gpus=8
    echo "Ray cluster started on the head node."
fi

pkill -f ${REWARD_MODEL}
nohup python -m openrlhf.cli.${REWARD_MODEL} --data_path $DATA_PATH --reward_pretrain $TOKENIZER_PATH --log_file results/$SAVE_MODEL_NAME/server/sampling.jsonl --port ${PORT} > $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log 2>&1 &
echo $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log 

if [ "$NODE_RANK" = "0" ]; then
ray job submit --address="http://127.0.0.1:8266" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain ${TOKENIZER_PATH} \
   --remote_rm_url http://localhost:${PORT}/get_reward \
   --save_path results/$SAVE_MODEL_NAME \
   --ckpt_path results/$SAVE_MODEL_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator group_norm \
   --max_samples ${MAX_SAMPLES} \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len $PROMPT_MAX_LENGTH \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate $LR \
   --critic_learning_rate 9e-6 \
   --init_kl_coef $KL \
   --prompt_data $DATA_PATH \
   --input_key messages \
   --apply_chat_template \
   --packing_samples \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10 \
   --use_wandb ${wandb_token} \
   --wandb_run_name $SAVE_MODEL_NAME \
   --vllm_sync_backend nccl \
   --max_ckpt_num 20 \
   --group_method $GROUP_METHOD \
   --use_length_reward_in_efficiency \
   --temperature $TEMP \
   --overlap_comm   
fi