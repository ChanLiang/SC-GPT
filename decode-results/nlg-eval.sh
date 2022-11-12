# for suffix in normal_ord pos_ord neg_ord lex_pos_ord lex_neg_ord pos_maj3 pos_maj10 neg_maj3 neg_maj10
# for suffix in lex_neg lex_maj3 lex_maj10 lex_nmaj3 lex_nmaj10
# do
# echo $suffix >> 905-res
# nlg-eval --hypothesis=hyp-${suffix} --references=ref >> 905-res
# done

suffix=$1
nlg-eval --hypothesis=hyp-${suffix} --references=ref
