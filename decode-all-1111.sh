MODEL_SAVE_PATH='./persona-output'

# for suffix in lex_neg lex_maj3 lex_maj10 lex_nmaj3 lex_nmaj10
# for suffix in lex_nmaj10
# for suffix in single_pos
# for suffix in multi_pos

# for suffix in normal pos neg
for suffix in normal pos neg lex_pos lex_neg
do
# export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=3
python generate.py \
--model_type=gpt2 \
--model_name_or_path $MODEL_SAVE_PATH \
--num_samples 1 \
--temperature 0.8 \
--input_file ../data/persona/personachat_test_${suffix}_order.txt \
--top_k 10 \
--top_p 0.9 \
--output_file decode-results-full/dialogpt-ft-results-${suffix}-top10_top0.9_T0.8.json \
--length 20
done