CUDA_VISIBLE_DEVICES=0 python3 PIECER_main.py \
    --with_cuda \
    --batch_size 4 \
    --acc_batch 1 \
    --data_ratio 1.0 \
    --epochs 4 \
    --warmup_ratio 0.06 \
    --lr 0.0005 \
    --ptm_lr 0.00001 \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 0.000001 \
    --grad_clip 5.0 \
    --weight_decay 0.01 \
    --d_model 768 \
    --dropout 0.1 \
    --char_dim 200 \
    --use_ema \
    --ema_decay 0.9999 \
    --print_freq 500 \
    --model RoBERTa \
    --ptm_dir ./data/original/RoBERTa/RoBERTa_base/ \
    --ent_emb_file ./data/processed/ReCoRD/transe10k_stemmed_ent_emb.pkl \
    --ent2id_file ./data/processed/ReCoRD/transe10k_stemmed_ent2id.json \
    --use_ent_emb \
    --use_kg_gcn \
    --gcn_pos emb \
    --gcn_num_layer 3 \
    --after_matching \
    --processed_data \
    --save_prefix roberta_base_PIECER