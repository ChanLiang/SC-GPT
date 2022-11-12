 export CUDA_VISIBLE_DEVICES=0
 python preprocess_multitask.py --dataset_type convai2 \
 --testset /misc/kfdata01/kf_grp/lchen/data/persona/parlai/parl.ai/downloads/personachat/personachat/train_self_original_wo_candidates.txt \
 --split train \
 --encoder_model_name_or_path ../bert-base-models \
 --max_source_length 384 \
 --max_target_length 32
