```sh
python3 process_fasta.py
```

```sh
seed=100
prefix=P8Exp2100_129tracks_34P
model_path=../../seq2exp/best_model_dirs/${prefix}_TrunkFrozen_PaddyHead_best_model_dir/seed${seed}_model_best.h5
yaml=../../seq2exp/transfer_CE_PaddyHead.yaml

model_path=../../seq2exp/tf_exps/${seed}_${prefix}_train_transfer_out/model_best.h5
yaml=../../seq2exp/tf_exps/${seed}_${prefix}_train_transfer_out/params.yaml 

paddy_grad_fa.py --model_path $model_path \
    $yaml \
    -o ${prefix}_${seed}.h5 \
    WetExp_genes.fa
motif_of_RtSe_plot.py --output_format pdf --h5_file ${prefix}_${seed}.h5 \
    --output_dir ${prefix}_${seed}_RtSe_plots --motif_json motif_info.json \
    --fasta_file WetExp_genes.fa --tissue_dict 23tissues_modified_dict.json
```

```sh
prefix=P8Exp2100_129tracks_34P
seed=100
model_path=../../seq2exp/tf_exps/${seed}_${prefix}_train_transfer_out/model_best.h5
yaml=../../seq2exp/tf_exps/${seed}_${prefix}_train_transfer_out/params.yaml 
paddy_ig_fa.py --model_path $model_path \
    $yaml \
    -o ${prefix}_${seed}_igs.h5 \
    WetExp_genes.fa
motif_ig_RtSe_plot.py --output_format png --h5_file ${prefix}_${seed}_igs.h5 \
    --output_dir ${prefix}_${seed}_igs_RtSe_plots --motif_json motif_info.json \
    --fasta_file WetExp_genes.fa --tissue_dict 23tissues_modified_dict.json

```
