#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="/root/miniconda3/bin/python"

export PYTHONPATH="/home/donny/torchcv-seg":$PYTHONPATH

cd ../../

DATA_DIR="/home/donny/DataSet/ADE20K"
BACKBONE="deepbase_resnet50_dilated8"
MODEL_NAME="pspnet"
LOSS_TYPE="fs_auxce_loss"
CHECKPOINTS_NAME="fs_res50_pspnet_ade20k_seg"$2
PRETRAINED_MODEL="./pretrained_model/resnet50-imagenet.pth"
MAX_ITERS=150000

LOG_DIR="./log/ade20k/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [ ! -d ${LOG_DIR} ]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_pspnet_ade20k_seg.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} > ${LOG_FILE} 2>&1

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_pspnet_ade20k_seg.json --drop_last y \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --gpu 0 1 2 3 \
                       --resume_continue y --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL}  >> ${LOG_FILE} 2>&1

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_pspnet_ade20k_seg.json \
                       --phase debug --gpu 0 --log_to_file n > ${LOG_FILE} 2>&1

elif [ "$1"x == "val"x ]; then
  ${PYTHON} -u main.py --hypes hypes/ade20k/fs_pspnet_ade20k_seg.json \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --log_to_file n --out_dir val >> ${LOG_FILE} 2>&1
  cd val/scripts
  ${PYTHON} -u ade20k_evaluator.py --hypes_file ../../hypes/ade20k/fs_pspnet_ade20k_seg.json \
                                   --pred_dir ../results/ade20k/test_dir/${CHECKPOINTS_NAME}/val/label \
                                   --gt_dir ${DATA_DIR}/val/label  >> "../../"${LOG_FILE} 2>&1
else
  echo "$1"x" is invalid..."
fi
