MODEL_SAVE_PATH='./persona-output'

export CUDA_VISIBLE_DEVICES=2
python generate.py \
--model_type=gpt2 \
--model_name_or_path $MODEL_SAVE_PATH \
--num_samples 5 \
--input_file='../data/persona/ordered_data/personachat_test_120_lex_order.txt' \
--top_k 5 \
--output_file='dialogpt-ft-results-lex.json' \
--length 80