# Set environment variables as in your original script
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0
# set node rank to 0 since we are not using ray
NODE_RANK=0

# Your wandb token
wandb_token=$WANDB_TOKEN

# Paths for training data and backbone model
DATA_PATH=/mnt/longcontext/models/siyuan/rl_datasets/STILL-3-Preview-RL-Data
# TOKENIZER_PATH=/mnt/longcontext/models/siyuan/llama3/DeepSeek-R1-Distill-Qwen-7B
TOKENIZER_PATH=/mnt/longcontext/models/siyuan/llama3/llama3-8b-Instruct

# Hyperparameters
N_SAMPLES=8
EPISODE=1
WARMUP=0.0
TBS=512
RBS=128
KL=0.001
LR=2e-6
MAX_LENGTH=16384
PORT=1278
TEMP=0.6
# REWARD_MODEL=server_dpsk_tuple
REWARD_MODEL=server_llama3_reward
# TODO: write a new reward model
SAVE_MODEL_NAME=test_llama3-8b-rm1-1-2-grpo-len_${MAX_LENGTH}-tbs_${TBS}-rbs_${RBS}-sample_$N_SAMPLES-kl_${KL}-warmup_${WARMUP}-ep_${EPISODE}-plr_${LR}-temp$TEMP-30k
GROUP_METHOD=normal
LOG_BASE=log

# Create necessary directories
mkdir -p results/$SAVE_MODEL_NAME
mkdir -p results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/

# start reward
nohup python -m openrlhf.cli.${REWARD_MODEL} --data_path $DATA_PATH --reward_pretrain $TOKENIZER_PATH --log_file results/$SAVE_MODEL_NAME/server/sampling.jsonl --port ${PORT} > $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log 2>&1 &

# ===========================
# test the reward model server only
# ===========================
# python -m openrlhf.cli.server_llama3_reward --data_path /mnt/longcontext/models/siyuan/rl_datasets/STILL-3-Preview-RL-Data --reward_pretrain /mnt/longcontext/models/siyuan/llama3/llama3-8b-Instruct --log_file test_reward_model_llama3 --port 1278

# Start the training with Deepspeed instead of Ray
deepspeed --module openrlhf.cli.train_ppo \
    --pretrain ${TOKENIZER_PATH} \
    --reward_pretrain ${TOKENIZER_PATH} \
    --save_path results/$SAVE_MODEL_NAME \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 1 \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size ${RBS} \
    --max_samples 1000 \
    --max_epochs 1 \
    --num_episodes ${EPISODE} \
    --lr_warmup_ratio ${WARMUP} \
    --n_samples_per_prompt $N_SAMPLES \
    --prompt_max_len 4096 \
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
    --use_wandb $wandb_token \
    --wandb_run_name $SAVE_MODEL_NAME \
    --vllm_sync_backend nccl \
    --max_ckpt_num 20 \
    --group_method $GROUP_METHOD \
    --use_length_reward_in_efficiency \
    --temperature $TEMP \
    --overlap_comm > $LOG_BASE/server/$SAVE_MODEL_NAME-node0.log 2>&1 &
echo "Training started with Deepspeed. Logs: $LOG_BASE/server/$SAVE_MODEL_NAME-node0.log"