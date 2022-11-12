
# suffix='multi_pos'
# suffix='single_pos'
# for suffix in 'normal_ord' 'pos_ord', 'neg_ord'
# for suffix in 'lex_pos_ord' 'lex_neg_ord' 'pos_maj3'
# for suffix in 'pos_maj10' 'neg_maj3' 'neg_maj10'
for suffix in 'single_pos' 'multi_pos'
do


# 1. parse hyp and ref
cd decode-results
python get_hyp.py $suffix

# 2. evaluate nlp metrics
source /misc/kfdata01/kf_grp/lchen/anaconda3/etc/profile.d/conda.sh
conda activate nlg-eval
bash nlg-eval.sh $suffix > eval-res/$suffix

done