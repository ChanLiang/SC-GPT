MODEL_SAVE_PATH='./persona-output'
# PRE_TRAINED_MODEL_PATH='microsoft/DialoGPT-medium'
PRE_TRAINED_MODEL_PATH='../dialogpt-models'
EPOCH=10
LR=5e-5

#export CUDA_VISIBLE_DEVICES=2,3
# python train.py  \
python -m torch.distributed.launch --nproc_per_node=4 train.py \
--output_dir=$MODEL_SAVE_PATH \
--model_type=gpt2 \
--model_name_or_path=$PRE_TRAINED_MODEL_PATH \
--do_train --do_eval \
--eval_data_file='../data/persona/personachat_train.txt' \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--num_train_epochs $EPOCH \
--learning_rate $LR \
--overwrite_cache \
--use_tokenize \
--train_data_file='../data/persona/personachat_valid.txt' \
--overwrite_output_dir

# --per_gpu_train_batch_size 8 \
# --per_gpu_train_valid_size 8 \
# --gradient_accumulation_steps 1 \
# --max_seq 256 \
# --save_total_limit 3 \
