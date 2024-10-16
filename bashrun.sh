#!/bin/bash

# model: ResNet, LinearNet_rcv1
MODEL='ResNet'

# dataset: CIFAR10, CIFAR100, RCV1
DATASET='CIFAR100'

LOGDIR_OUT='./log'
LOGDIR='./runs/'
if [ ! -d $LOGDIR_OUT ]; then
  mkdir -p $LOGDIR_OUT
fi
if [ ! -d ${LOGDIR_OUT}/${MODEL}_${DATASET} ]; then
  mkdir -p ${LOGDIR_OUT}/${MODEL}_${DATASET}
fi

NUM_WORKERS=16
BATCH_SIZE=16
EPOCH=200
SEED=42
# SEED_list='1 19 42 44 80'

LR=0.01

DELAY=4
JOB_NAME=${LOGDIR_OUT}/${MODEL}_${DATASET}/log_worker-${NUM_WORKERS}_${DELAY}delay_bs-${BATCH_SIZE}_lr-${LR}_seed-${SEED}
echo $JOB_NAME
CUDA_VISIBLE_DEVICES=0 python -u server.py --model $MODEL --cuda-ps --batch-size $BATCH_SIZE \
                                    --dataset $DATASET --delay $DELAY --logdir $LOGDIR \
                                    --num-workers $NUM_WORKERS --lr $LR --num-epochs $EPOCH --seed $SEED\
                                    > ${JOB_NAME}.out 2>&1 &
wait

DELAY=8
JOB_NAME=${LOGDIR_OUT}/${MODEL}_${DATASET}/log_worker-${NUM_WORKERS}_${DELAY}delay_bs-${BATCH_SIZE}_lr-${LR}_seed-${SEED}
echo $JOB_NAME
CUDA_VISIBLE_DEVICES=1 python -u server.py --model $MODEL --cuda-ps --batch-size $BATCH_SIZE \
                                    --dataset $DATASET --delay $DELAY --logdir $LOGDIR \
                                    --num-workers $NUM_WORKERS --lr $LR --num-epochs $EPOCH --seed $SEED\
                                    > ${JOB_NAME}.out 2>&1 &
wait

DELAY=16
JOB_NAME=${LOGDIR_OUT}/${MODEL}_${DATASET}/log_worker-${NUM_WORKERS}_${DELAY}delay_bs-${BATCH_SIZE}_lr-${LR}_seed-${SEED}
echo $JOB_NAME
CUDA_VISIBLE_DEVICES=2 python -u server.py --model $MODEL --cuda-ps --batch-size $BATCH_SIZE \
                                    --dataset $DATASET --delay $DELAY --logdir $LOGDIR \
                                    --num-workers $NUM_WORKERS --lr $LR --num-epochs $EPOCH --seed $SEED\
                                    > ${JOB_NAME}.out 2>&1 &
wait

DELAY=32
JOB_NAME=${LOGDIR_OUT}/${MODEL}_${DATASET}/log_worker-${NUM_WORKERS}_${DELAY}delay_bs-${BATCH_SIZE}_lr-${LR}_seed-${SEED}
echo $JOB_NAME
CUDA_VISIBLE_DEVICES=3 python -u server.py --model $MODEL --cuda-ps --batch-size $BATCH_SIZE \
                                    --dataset $DATASET --delay $DELAY --logdir $LOGDIR \
                                    --num-workers $NUM_WORKERS --lr $LR --num-epochs $EPOCH --seed $SEED\
                                    > ${JOB_NAME}.out 2>&1 &

wait

echo 'finish!'

