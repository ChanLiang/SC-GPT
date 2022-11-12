MODEL_SAVE_PATH='./persona-output-multitask/checkpoint-10000'

export CUDA_VISIBLE_DEVICES=0
python generate.py \
--model_type=gpt2 \
--model_name_or_path $MODEL_SAVE_PATH \
--num_samples 1 \
--input_file='data/ConvAI2/convai2_tokenized_task1/test.txt' \
--top_k 5 \
--output_file='decode-results/multitask.json' \
--length 38