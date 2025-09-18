CUDA_VISIBLE_DEVICES=1 python main_train.py --image_dir ../dataset/mimic_cxr/images/ --ann_path data/mimic_cxr/mimic_annotation.json --dataset_name mimic_cxr --gen_max_len 150 --gen_min_len 100 --batch_size 16 --epochs 10 --save_dir results/ --seed 456789 --init_lr 5e-5 --min_lr 5e-6 --warmup_lr 5e-7 --weight_decay 0.05 --warmup_steps 2000 --cls_weight 4 --clip_k 21 --beam_size 3      

