MODEL_SAVE_PATH='./output'
PRE_TRAINED_MODEL_PATH='./scgpt'
EPOCH=3
LR=5e-5

export CUDA_VISIBLE_DEVICES=3
python train.py  \
--output_dir=$MODEL_SAVE_PATH \
--model_type=gpt2 \
--model_name_or_path=$PRE_TRAINED_MODEL_PATH \
--do_train --do_eval \
--eval_data_file='./data/restaurant/train.txt' \
--per_gpu_train_batch_size 1 \
--num_train_epochs $EPOCH \
--learning_rate $LR \
--overwrite_cache \
--use_tokenize \
--train_data_file='./data/restaurant/train.txt' \
--overwrite_output_dir
