#! /bin/bash

# Runs the tiny parameter model
# DATA_PATH=GPT2/c4_en_partial_gpt2_text_document
# CHECKPOINT_PATH=GPT2

DATA_PATH=data/meg-gpt2-oscar-en-10k_text_document
CHECKPOINT_PATH=checkpoints/tiny/gpt2
VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt


GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 2 \
       --hidden-size 128 \
       --num-attention-heads 4 \
       --micro-batch-size 4 \
       --global-batch-size 4 \
       --seq-length 256 \
       --max-position-embeddings 256 \
       --train-iters 10000 \
       --lr-decay-iters 5000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       # --tensorboard-dir GPT2

#        --vocab-file GPT2/gpt2-vocab.json \
#        --merge-file GPT2/gpt2-merges.txt \

#  --tokenizer-type PretrainedFromHF \
#  --tokenizer-name-or-path t5-small \