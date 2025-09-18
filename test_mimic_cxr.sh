CUDA_VISIBLE_DEVICES=1, python main_test.py \
--n_gpu 1 \
--image_dir ../dataset/mimic_cxr/images/ \
--ann_path data/mimic_cxr/mimic_annotation.json \
--dataset_name mimic_cxr \
--gen_max_len 150 \
--gen_min_len 100 \
--batch_size 16 \
--save_dir results/ \
--seed 456789 \
--clip_k 21 \
--beam_size 3 \
--load_pretrained results/model_best_t_mid_gk3.23.pth
# --load_pretrained results/model_best.pth
