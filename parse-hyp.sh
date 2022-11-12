
for suffix in 'normal' 'pos' 'neg' 'lex_neg'
do

# 1. parse hyp and ref
python get_hyp.py $suffix

# 2. evaluate nlp metrics
# source /misc/kfdata01/kf_grp/lchen/anaconda3/etc/profile.d/conda.sh
# conda activate nlg-eval
# bash nlg-eval.sh $suffix > eval-res/$suffix

done