export LOG_DIR=./logs
export DATA_DIR=./data/original
export TASK_NAME=SST-2
export MODEL=begin_with_sentiment_and_downline
export DATA_RATIO=1
export NLPEXP=./output
mkdir -p $DATA_DIR/$TASK_NAME/$DATA_RATIO/$MODEL
mkdir -p $LOG_DIR/$TASK_NAME/$DATA_RATIO/
touch $LOG_DIR/$TASK_NAME/$DATA_RATIO/$MODEL
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --model_name_or_path $MODEL \
  --data_cached_dir $DATA_DIR/$TASK_NAME/$DATA_RATIO/$MODEL \
  --task_name $TASK_NAME \
  --load_best_model_at_end \
  --disable_tqdm True \
  --metric_for_best_model acc \
  --do_train \
  --do_eval \
  --seed 42 \
  --evaluation_strategy epoch \
  --data_dir $DATA_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --eval_steps 500 \
  --overwrite_output_dir \
  --data_ratio $DATA_RATIO \
  --output_dir $NLPEXP/$TASK_NAME/$DATA_RATIO/$MODEL
