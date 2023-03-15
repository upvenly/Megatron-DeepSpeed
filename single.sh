#!/bin/bash
export NCCL_SOCKET_IFNAME=ib0
export NCCL_PXN_DISABLE=0
export NCCL_IB_HCA=mlx5_0  #0号网卡
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_FIND_MODE=3

export MIOPEN_USER_DB_PATH=/tmp/miopen-udb
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache

mkdir -p log_wwl
now=$(date +"%Y%m%d_%H%M%S")

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE


MODEL_NAME=gpt3_175B
DATA_OUTPUT_PATH=./
LOGS_PATH=$DATA_OUTPUT_PATH/logs
CHECKPOINT_PATH=checkpoints/$MODEL_NAME

DATA_PATH="/public/home/platform/wwl/Megatron-DeepSpeed/data/meg-gpt2-oscar-en-10k_text_document"
TENSORBOARD_PATH=output_dir/tensorboard/$MODEL_NAME
CODECARBON_PATH=output_dir/codecarbon/$MODEL_NAME


SAVE_INTERVAL=250

TP_SIZE=4
PP_SIZE=88

NHIDDEN=12352 #12352
NLAYERS=88
NHEADS=64
SEQ_LEN=2048


ZERO_STAGE=1
config_json="./${MODEL_NAME}_ds_config.json"



MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1 #5760
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
export CMD=" \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size 1 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples 360000 \
    --loss-scale 12 \
    --clip-grad 1.0 \
    --fp16 \
    --checkpoint-activations \
    --seed 42
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --exit-duration-in-mins 1190 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file /public/home/platform/wwl/Megatron-DeepSpeed/data/gpt2-vocab.json \
    --merge-file /public/home/platform/wwl/Megatron-DeepSpeed/data/gpt2-merges.txt \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 40 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "
#  APP="deepspeed --num_gpus 1 /work/home/jsyadmin/.conda/envs/ldk-h/bin/python3.7 -u `pwd`/pretrain_gpt.py \

APP="python -u `pwd`/pretrain_gpt.py \
    --rank ${RANK} \
    --world_size ${WORLD_SIZE} \
    --dist_url tcp://${1}:34566 \
    --num-workers 2 \
    ${CMD} \
    "
#echo ${APP}



case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  #echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP}
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP}

  #echo GLOO_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP} 
  #GLOO_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  #echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=1 --membind=1 ${APP}
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
  #echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=2 --membind=2 ${APP} 
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
  #echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=3 --membind=3 ${APP}
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac

