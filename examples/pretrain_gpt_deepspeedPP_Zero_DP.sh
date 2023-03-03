#! /bin/bash

# Runs the "345M" parameter model

# RANK=0
# WORLD_SIZE=1

# DATA_PATH=<Specify path and file prefix>_text_document
# CHECKPOINT_PATH=<Specify path>

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

DATA_PATH=data/meg-gpt2-oscar-en-10k_text_document
CHECKPOINT_PATH=checkpoints/345/gpt2-deepspeedPP_ZeroDP
VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt
TENSORBOARD_PATH=output_dir/tensorboard
CODECARBON_PATH=output_dir/codecarbon


if [ -d "$CHECKPOINT_PATH" ]
then
    rm -rf $CHECKPOINT_PATH
fi


N_GPUS=8
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=64
TP_SIZE=2
PP_SIZE=2
ZERO_STAGE=1
SAVE_INTERVAL=1000


GPT_ARGS="--num-layers 24 \
      --hidden-size 1024 \
      --num-attention-heads 16 \
      --micro-batch-size 8 \
      --global-batch-size 64 \
      --seq-length 1024 \
      --max-position-embeddings 1024 \
      --train-iters 1000 \
      --lr-decay-iters 320000 \
      --save $CHECKPOINT_PATH \
      --load $CHECKPOINT_PATH \
      --data-path $DATA_PATH \
      --vocab-file $VOCAB_FILE \
      --merge-file $MERGE_FILE \
      --data-impl mmap \
      --split 949,50,1 \
      --distributed-backend nccl \
      --lr 0.00015 \
      --lr-decay-style cosine \
      --min-lr 1.0e-5 \
      --weight-decay 1e-2 \
      --clip-grad 1.0 \
      --lr-warmup-fraction .01 \
      --checkpoint-activations \
      --log-interval 100 \
      --save-interval 10000 \
      --eval-interval 1000 \
      --eval-iters 10 \
      --fp16"

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

#    --codecarbon-dir $CODECARBON_PATH \
DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "



config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

ALL_ARGS="$GPT_ARGS $OUTPUT_ARGS $DATA_ARGS $DEEPSPEED_ARGS"

# if you can't stand pt-1.9 launcher noise
export LOGLEVEL=WARNING

LAUNCHER="deepspeed --num_gpus $N_GPUS"
export CMD=" \
    $LAUNCHER pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    $ALL_ARGS \
    "

echo $CMD

$CMD 2>&1 | tee log/gpt2-deepspeedPP-zereDP.log.$now

