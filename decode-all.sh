MODEL_SAVE_PATH='./persona-output'

# for suffix in lex_neg lex_maj3 lex_maj10 lex_nmaj3 lex_nmaj10
# for suffix in lex_nmaj10
# for suffix in single_pos
for suffix in multi_pos
do
export CUDA_VISIBLE_DEVICES=3
python generate.py \
--model_type=gpt2 \
--model_name_or_path $MODEL_SAVE_PATH \
--num_samples 5 \
--input_file ../data/persona/personachat_test_120_${suffix}_order.txt \
--top_k 5 \
--output_file decode-results/dialogpt-ft-results-${suffix}.json \
--length 40
done