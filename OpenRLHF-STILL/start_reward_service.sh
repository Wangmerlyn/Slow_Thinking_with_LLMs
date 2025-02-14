

NODE_RANK=$1

# export TORCH_HOME=/opt/aps/workdir
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0

# Your wandb token
wandb_token=$WANDB_TOKEN
sudo rm -rf ~/.netrc

# Path of training data
# DATA_PATH=/mnt/longcontext/models/siyuan/rl_datasets/STILL-3-preview-RL-Data-1k
DATA_PATH=/mnt/longcontext/models/siyuan/rl_datasets/longcontext_train_30k/train.jsonl

# Path of backbone model(DeepSeek-R1-Distill-Qwen-1.5B)
TOKENIZER_PATH=/mnt/longcontext/models/siyuan/llama3/llama-3.1-8B-instruct


MAX_SAMPLES=256
N_SAMPLES=32
EPISODE=1
WARMUP=0.0
TBS=512
RBS=128
KL=0.001
LR=2e-6
MAX_LENGTH=16384
PORT=1278
TEMP=0.6
# REWARD_MODEL=server_false-1_true1_unknown-1-repeat-single
# REWARD_MODEL=server_dpsk_tuple
REWARD_MODEL=server_llama3_reward
SAVE_MODEL_NAME=test-llama31-8b-rm1-1-2-grpo-len_${MAX_LENGTH-}tbs_${TBS}-rbs_${RBS}-sample_$N_SAMPLES-kl_${KL}-warmup_${WARMUP}-ep_${EPISODE}-plr_${LR}-temp$TEMP-30k

GROUP_METHOD=normal

LOG_BASE=log

mkdir -p results/$SAVE_MODEL_NAME
mkdir -p results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/

pkill -f ${REWARD_MODEL}
nohup python -m openrlhf.cli.${REWARD_MODEL} --data_path $DATA_PATH --reward_pretrain $TOKENIZER_PATH --log_file results/$SAVE_MODEL_NAME/server/sampling.jsonl --port ${PORT} > $LOG_BASE/server/re_test.log 2>&1 &
echo $LOG_BASE/server/re_test.log