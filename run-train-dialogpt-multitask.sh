MODEL_SAVE_PATH='./persona-output-multitask'
PRE_TRAINED_MODEL_PATH='../dialogpt-models'
EPOCH=10
LR=5e-5

export CUDA_VISIBLE_DEVICES=0,1
# python train.py  \
python -m torch.distributed.launch --nproc_per_node=2 train.py \
--output_dir=$MODEL_SAVE_PATH \
--model_type=gpt2 \
--model_name_or_path=$PRE_TRAINED_MODEL_PATH \
--do_train \
--do_eval \
--eval_all_checkpoints \
--eval_data_file='data/ConvAI2/convai2_tokenized_task1/test.txt' \
--gradient_accumulation_steps 2 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 8 \
--num_train_epochs $EPOCH \
--learning_rate $LR \
--max_seq 128 \
--overwrite_cache \
--use_tokenize \
--train_data_file='data/ConvAI2/convai2_tokenized_task_mix/train_shuf.txt' \
--overwrite_output_dir 1>log/train/916/res 2>log/train/916/err 

# --per_gpu_train_batch_size 8 \
# --per_gpu_train_valid_size 8 \
# --gradient_accumulation_steps 1 \
# --max_seq 256 \
# --save_total_limit 3 \
