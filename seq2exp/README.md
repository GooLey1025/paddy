```shell

hound_data.py -l 32768  --local -p 16 -o 49169_mseqs -w 32 NumChr.Rice_MSUv7.fa Nip8_106targets.txt --st
cp sequences_limited.bed 49169_mseqs/sequences.bed
hound_data.py -l 32768  --local -p 16 -o 49169_mseqs -w 32 NumChr.Rice_MSUv7.fa Nip8_106targets.txt --restart
hound_train.py -o test_train_out -l test_log_dir 23ave.params_micro_106targets.json 49169_mseqs/

# Transfer Learning
./se_data_transfer.py NumChr.Rice_MSUv7.fa Nip_ATGsite_UD16K_Bed.xlsx -o transfer_data_out -p 32 -t Nip8_106targets.txt
json_to_yaml.py 23ave.params_micro_106targets.json -o 23ave.                      params_micro_106targets_transfer.yaml
# manually add `transfer and head params` in *_transfer.yaml

prefix=400
paddy_transfer.py --seed $prefix -g 0 -o ${prefix}_train_transfer_out -l ${prefix}_log_transfer_out --restore borzoi_rp_models/ChromatinModel_R1_tmp180.h5 --trunk 23ave.params_micro_106targets_transfer.yaml transfer_data_out > ${prefix}.train.log

# GRIDS=grid_search_20250704_night
# train_grid.py  -s "paddy_transfer.py" -a " " --output_dir experiments/$GRIDS --seeds 1 100 200 300 400 -p 1

# Just predict based on one best model
se_predict.py -m ${prefix}_train_transfer_out/model_best.h5 -f NumChr.Rice_MSUv7.fa -b sequences_to_predict.bed --params_file 23ave.params_micro_106targets_transfer.yaml -o preds --save_fasta
# If want to compare with different seeds
mkdir -p model_dir
# move best_models whose name are seed100_*.h5 to the model_dir
se_predict.py -m model_dir -f NumChr.Rice_MSUv7.fa -b sequences_to_predict.bed --params_file 23ave.params_micro_106targets_transfer.yaml -o preds --save_fasta

```
