
## Train Borzoi(pretrained) Model

```sh
# split datasets
hound_data.py -l 32768  --local -p 16 -o full_mseqs -w 32 NumChr.Rice_MSUv7.fa Nip8_106targets.txt --st
```
Here assign chr1 as valid set, chr2 as test set, while others are train sets.
`xlsx` file contains columns like his
```xlsx
Type    GeneID  Chrom   Start   End Flag
ATG_UD16K LOC_Os01g01040    1   215 32983   valid
ATG_UD16K   LOC_Os01g01050  1   6848   39616   valid
...
ATG_UD16K   LOC_Os02g01060  2   16724   49492   test
...
```
Transfrom xlsx into bed file
```sh
xlsx_to_bed.py -o full_mseqs/sequences.bed --chrom 2 --start 3 --end 4 --flag 5 --format bed4 Nip_MidGene_Win10k_ATGUD16K_Bed_Combined.xlsx
```
Generate tfrecord files and train borzoi model (Sequence-to-Coverage model, SC model)
```sh
hound_data.py -l 32768  --local -p 16 -o full_mseqs -w 32 NumChr.Rice_MSUv7.fa Nip8_106targets.txt --restart
hound_train.py -o test_train_out -l test_log_dir 23ave.params_micro_106targets.json full_mseqs
```
## Train *Sequence to Expression*(SE) model

### (Optional) Directly train SE model 
Borzoi model architecture, but modifying the final head (in yaml) to produce a 1D output vector matching the shape of the target expression labels.
Its model performance (predicting gene expression) is not most excellent.

Prepare tfrecords datasets
```sh
se_data.py NumChr.Rice_MSUv7.fa Nip_ATGsite_UD16K_Bed.xlsx -o transfer_data_out -p 32 -t Nip8_106targets.txt
``` 
Train
```sh
se_train.py -o se_train_out -l se_log_dir 23ave.params_micro_106targets.yaml transfer_data_out
```
### Transfer Learning
Instead of fine-tuning the Borzoi model end-to-end, we extract intermediate embeddings`(before head)` and train a new downstream model for expression prediction, as this achieves better Pearson’s R.

Make datasets used for transfer learning.
```sh
se_data_transfer.py all_34samples_trainset.ATG_UD16K.seq2exp --valid_chrom Chr1 --test_chrom Chr2 -o new_ATG_UpDown16K_transfer_data_out -p 16  
# or select samples
se_data_transfer.py all_34samples_trainset.ATG_UD16K.seq2exp --valid_chrom Chr1 --test_chrom Chr2 -o 6P_new_ATG_UpDown16K_transfer_data_out -p 16 --filter_ids 6P.txt
```
```sh
json_to_yaml.py 23ave.params_micro_106targets.json -o 23ave.params_micro_106targets_transfer.yaml
# then manually add `transfer and head params` in transfer.yaml
```
After converting json to yaml, you could edit transfer.yaml.
- Assign `transfer mode` with `linear` to freeze trunk weights, using intermediate embeddings.
- Assign `transfer mode` with `full` to make all weights trainable, finetuning the model.
- Assign `transfer mode` with `adapter` to finetune in specific ways.



#### Clarification of `34Prp2_23tracks_34P`

`34P`: A Sequence-to-Coverage (SC, Pretrained) model traind using data from a total of 34 genomes

`rp2`: A replicate model, meaning the SC model was trained using a different random seed (replicate 2) to assess training variability.

`23tracks`: The SC model was trained with 23 RNA-seq tracks as output targets (labels).

`34P`: Refers to the Transfer model (SE model), which uses embedding from the above SC model trained on 34 genomes as input features.

Train model across different seeds.
```sh
mkdir -p tf_exps
prefix=34Prp2_23tracks_34P
seed=100 # 200 300 400
nohup paddy_transfer.py --seed $seed -o tf_exps/${seed}_${prefix}_train_transfer_out -l tf_exps/${seed}_${prefix}_log_transfer_out --restore SC_models/34P_Borzoi_Seq2RNAseqTracks_rp2_model_best.h5 --trunk transfer_CE_PaddyHead.yaml new_ATG_UpDown16K_transfer_data_out > tf_exps/${seed}_${prefix}.train.log &

prefix=P8Exp0100_129tracks_34P
seed=100
nohup paddy_transfer.py --seed $seed -o tf_exps/${seed}_${prefix}_train_transfer_out -l tf_exps/${seed}_${prefix}_log_transfer_out --restore 129_pretrain_model_ablations/exp0_100/model_best.h5 --trunk 129_pretrain_model_ablations/exp0_100/exp0_100.yaml new_ATG_UpDown16K_transfer_data_out > tf_exps/${seed}_${prefix}.train.log &

prefix=P8Exp0100_129tracks_34Pft
seed=100
nohup paddy_transfer.py --seed $seed -o tf_exps/${seed}_${prefix}_train_transfer_out -l tf_exps/${seed}_${prefix}_log_transfer_out --restore 129_pretrain_model_ablations/exp0_100/model_best.h5 --trunk 129_pretrain_model_ablations/exp0_100/exp0_100_finetune.yaml new_ATG_UpDown16K_transfer_data_out > tf_exps/${seed}_${prefix}.train.log &

```
(Optional, not checked) If want to search hyperparameters of best model, do grid search experiments. 
```sh
# GRIDS=grid_search_20250704_night
# train_grid.py  -s "paddy_transfer.py" -a " " --output_dir experiments/$GRIDS --seeds 1 100 200 300 400 -p 1
```
Train model across different seeds.
```sh
mkdir -p MF_exps
prefix=34Prp1_23tracks_34P
seed=100
nohup paddy_train.py --pretrained SC_models/34P_Borzoi_Seq2RNAseqTracks_rp1_model_best.h5 --trunk -s ${seed} MultiFuse.yaml new_ATG_UpDown16K_transfer_data_out -o MF_exps/${seed}_${prefix}_train_transfer_out -l MF_exps/${seed}_${prefix}_log_transfer_out > MF_exps/${seed}_${prefix}.train.log &

nohup paddy_train.py --pretrained pretrain_model_ablations/exp0_100/model_best.h5 --trunk -s ${seed} 106tracks_MultiFuse.yaml new_ATG_UpDown16K_transfer_data_out -o MF_exps/${seed}_${prefix}_train_transfer_out -l MF_exps/${seed}_${prefix}_log_transfer_out > MF_exps/${seed}_${prefix}.train.log &
```
### Inference
```sh
# Just predict based on one best model
se_predict.py -m train_transfer_out/model_best.h5 -ref NumChr.Rice_MSUv7.fa -b sequences_to_predict.bed --params_file transfer_CE_PaddyHead.yaml -o preds --save_fasta

# If want to compare with different seeds
prefix=34Prp2_23tracks_6P
mkdir -p best_model_dirs/${prefix}_TrunkFrozen_PaddyHead_best_model_dir
# move best_models whose name are seed100_*.h5 to the model_dir
se_predict.py -m ${prefix}_TrunkFrozen_PaddyHead_best_model_dir -ref NumChr.Rice_MSUv7.fa -b sequences_to_predict.bed --params_file transfer_CE_PaddyHead.yaml -o ${prefix}_preds --plot

# or directly based on input fasta files
se_predict.py -m ${prefix}_TrunkFrozen_PaddyHead_best_model_dir -i IPA1_ATGud16k_RevComp.fa.20N_masked.fa --params_file transfer_CE_PaddyHead.yaml -o ism_preds --batch_size 64 
```
Evaluate across different models.
```sh
se_eval.py transfer_CE_PaddyHead.yaml ${prefix}_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5 new_ATG_UpDown16K_transfer_data_out -o eval_out

# run pipe multiple times
prefix=34Prp2_23tracks_34P
seed=100

nohup se_eval_pipe.sh transfer_CE_PaddyHead.yaml \
    best_model_dirs/${prefix}_TrunkFrozen_PaddyHead_best_model_dir/seed${seed}_model_best.h5 \
    new_ATG_UpDown16K_transfer_data_out \
    new_ATG_UpDown16K_transfer_data_out/unique_identifiers.txt \
    se_eval_exps_${seed}_${prefix} > se_eval_exps_${seed}_${prefix}.log &

seed=100
prefix=P8Exp0100_129tracks_34P$seed
nohup se_eval_pipe.sh 129_pretrain_model_ablations/exp0_100/exp0_100.yaml \
    best_model_dirs/${prefix}_TrunkFrozen_PaddyHead_best_model_dir/seed${seed}_model_best.h5 \
    new_ATG_UpDown16K_transfer_data_out \
    new_ATG_UpDown16K_transfer_data_out/unique_identifiers.txt \
    se_eval_exps_${prefix} > se_eval_exps_${prefix}.log &
format_metrics_tsv.py 
# combine all exps
> all_metrics_combined.tsv
head -n 1 $(find ./se_eval_exps_* -name "all_metrics.tsv" | head -n 1) > all_metrics_combined.tsv
find ./se_eval_exps_* -name "all_metrics.tsv" | while read f; do
    tail -n +2 "$f" >> all_metrics_combined.tsv
done
se_eval_pipe_plot.py -i all_metrics_combined.tsv
```
