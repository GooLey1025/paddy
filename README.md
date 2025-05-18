## Examples
```bash
# prepare label file
extract_label.py -x Nip_ATGsite_UD16K_Bed.xlsx -o 2_P8_Nip8_23tissues.Exp.csv -s 8

# generate tfrecords
paddy_data.py --h5_dir data/seqs_cov --output_dir test --label_file 2_P8_Nip8_23tissues.Exp --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1

# train
python3 paddy_train.py --train_dir test/tfrecords --output_dir test/output 

