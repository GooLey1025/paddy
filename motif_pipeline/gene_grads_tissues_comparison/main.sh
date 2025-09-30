#!/bin/bash

PREFIX=/data/xhhuanglab/gulei/projects/paddy
GENE_ID=LOC_Os08g39890 # IPA1: LOC_Os08g39890

GFF=$PREFIX/motif_pipeline/Rice_MSUv7.gff3
MODEL_PATH=$PREFIX/seq2exp/best_model_dirs/34P_23tracks_34P_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5
YAML=$PREFIX/seq2exp/transfer_CE_PaddyHead.yaml
FA=$PREFIX/seq2exp/NumChr.Rice_MSUv7.fa

grep "$GENE_ID" $GFF > $GENE_ID.gff3
paddy_grad_gene.py --model_path $MODEL_PATH --fa $FA --rc \
    --atg -o $GENE_ID.h5 $YAML $GENE_ID.gff3 

# explore_IPA1.ipynb as a template
cp explore_IPA1.ipynb explore_$GENE_ID.ipynb
