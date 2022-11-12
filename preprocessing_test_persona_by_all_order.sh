 export CUDA_VISIBLE_DEVICES=1
 python preprocess_batch_sort_persona_by_all_order.py --dataset_type convai2 \
 --testset /misc/kfdata01/kf_grp/lchen/data/persona/parlai/parl.ai/downloads/personachat/personachat/valid_self_original_wo_candidates.txt \
 --encoder_model_name_or_path ../bert-base-models \
 --max_source_length 384 \
 --max_target_length 32