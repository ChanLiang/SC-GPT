
# suffix='multi_pos'
# suffix='single_pos'
# for suffix in 'normal_ord' 'pos_ord', 'neg_ord'
# for suffix in 'lex_pos_ord' 'lex_neg_ord' 'pos_maj3'
# for suffix in 'pos_maj10' 'neg_maj3' 'neg_maj10'
for suffix in 'single_pos' 'multi_pos'
do

# 1. decode
# source env.sh
export CUDA_VISIBLE_DEVICES=3
python generate.py \
--model_type=gpt2 \
--model_name_or_path './persona-output-lex' \
--num_samples 1 \
--input_file ./data/ConvAI2/convai2_tokenized_${suffix}/test_${suffix}.txt \
--top_k 5 \
--output_file decode-results/dialogpt-ft-results-${suffix}.json \
--length 40

# 2. parse hyp and ref
cd decode-results
python get_hyp.py $suffix

# 3. evaluate nlp metrics
source /misc/kfdata01/kf_grp/lchen/anaconda3/etc/profile.d/conda.sh
conda activate nlg-eval
bash nlg-eval.sh $suffix > eval-res/$suffix

done