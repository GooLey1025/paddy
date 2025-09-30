PREFIX=/data/xhhuanglab/gulei/projects/paddy
MODEL_PATH=$PREFIX/seq2exp/best_model_dirs/34P_23tracks_34P_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5
FA=$PREFIX/seq2exp/NumChr.Rice_MSUv7.fa
YAML=$PREFIX/seq2exp/transfer_CE_PaddyHead.yaml
# GFF=$PREFIX/motif_pipeline/Rice_MSUv7.gff3

paddy_grad_gene.py --model_path $MODEL_PATH --fa $FA --rc \
    --atg -o all_genes_grads.h5 $YAML P8.gff --batch_size 4
