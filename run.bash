LOG_DIR="./"
NUM_WORKERS=16

TIMM=resnet50
EPOCHS=1
DATA_DIR=../input/hpa-single-cell-image-classification
INPUT_SIZE=512

# 1st SC-CAM round 0
EXT_MODE=0
ROUND_NB=0
python run_classification.py \
    --gpu 1 \
    --max_epochs $EPOCHS \
    --val_fold 0 \
    --batch_size 40 \
    --data_dir ${DATA_DIR} \
    --timm_model_name $TIMM \
    --benchmark  \
    --precision 16 \
    --optim_name sgd \
    --round_nb  ${ROUND_NB} \
    --aug_mode 1 \
    --default_root_dir ${LOG_DIR} \
    --num_workers ${NUM_WORKERS} \
    --num_inchannels 4 \
    --use_ext_data \
    --ext_data_mode ${EXT_MODE} \
    --input_size ${INPUT_SIZE} \
    --lr 0.5

ROUND_NB=1
CKPT_PATH=`ls ${LOG_DIR}/lightning_logs/version_0/checkpoints/epoch*`
SUB_LABEL_DIR="./save_uni"
mkdir $SUB_LABEL_DIR
python ./extract_feature.py \
    --weights $CKPT_PATH \
    --save_folder ${SUB_LABEL_DIR} \
    --ext_data_mode ${EXT_MODE} \
    --use_ext_data

python ./create_pseudo_label.py \
    --save_folder ${SUB_LABEL_DIR} \
    --for_round_nb $ROUND_NB \
    --is_scale_feature \
    --ext_data_mode ${EXT_MODE} \
    --use_ext_data


# 1st SC-CAM round 1
python run_classification.py \
    --gpu 1 \
    --max_epochs $EPOCHS \
    --val_fold 0 \
    --batch_size 40 \
    --data_dir ${DATA_DIR} \
    --timm_model_name $TIMM \
    --benchmark  \
    --precision 16 \
    --optim_name sgd \
    --round_nb  ${ROUND_NB} \
    --aug_mode 1 \
    --sub_label_dir ${SUB_LABEL_DIR} \
    --default_root_dir ${LOG_DIR} \
    --num_workers ${NUM_WORKERS} \
    --num_inchannels 4 \
    --use_ext_data \
    --ext_data_mode ${EXT_MODE} \
    --input_size ${INPUT_SIZE} \
    --lr 0.005


# psuedo label generation for 2nd stage
SEGM_DIR=$SUB_LABEL_DIR
python run_cam_infer.py \
    --yaml_path ./src/config/generate_psuedo_label.yaml \
    --sub_path ${SEGM_DIR}/psuedo_label.csv

python create_segm_labels.py \
    --save_folder ${SEGM_DIR} \
    --sub_csv ${SEGM_DIR}/psuedo_label.csv

# 2nd stage semantic segmentation
SEGM_TRESH=0.225
TIMM="resnet50"
INPUT_SIZE=512
python run_classification.py \
    --gpu 1 \
    --max_epochs $EPOCHS \
    --val_fold 0 \
    --batch_size 12 \
    --benchmark  \
    --precision 16 \
    --timm_model_name ${TIMM} \
    --optim_name sgd \
    --segm_label_dir $SEGM_DIR \
    --segm_thresh $SEGM_TRESH \
    --round_nb 0 \
    --num_workers $NUM_WORKERS \
    --default_root_dir $LOG_DIR \
    --num_inchannels 3 \
    --data_dir ${DATA_DIR} \
    --use_ext_data \
    --ext_data_mode ${EXT_MODE} \
    --input_size ${INPUT_SIZE} \
    --lr 0.25
