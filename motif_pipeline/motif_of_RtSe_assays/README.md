```sh
python3 process_fasta.py
```

```sh
seed=100
prefix=P8rp3_106tracks_34P
model_path=../../seq2exp/best_model_dirs/${prefix}_TrunkFrozen_PaddyHead_best_model_dir/seed${seed}_model_best.h5
yaml=../../seq2exp/transfer_CE_PaddyHead.yaml

paddy_grad_fa.py --model_path $model_path \
    $yaml \
    -o ${prefix}_${seed}.h5 \
    WetExp_genes.fa
motif_of_RtSe_plot.py --output_format pdf --h5_file ${prefix}_${seed}.h5 \
    --output_dir ${prefix}_${seed}_RtSe_plots --motif_json motif_info.json \
    --fasta_file WetExp_genes.fa --tissue_dict 23tissues_modified_dict.json
```
