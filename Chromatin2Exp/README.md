## Examples
```bash
cd Chromatin2Exp
chmod +x *.py

## 1.prepare data
# generate tfrecords via random split train_valid_test sets
./ce_data.py --h5_dir data/seqs_cov --output_dir 23tissues \
    --xlsx_file Nip_ATGsite_UD16K_Bed_new.xlsx \
    --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1
# or generate tfrecords via specify a specific chrom as the test or valid set, the rest are train set.
./ce_data.py --h5_dir data/seqs_cov --output_dir 23tissues \
    --xlsx_file Nip_ATGsite_UD16K_Bed_new.xlsx \
    --valid_chrom 11 --test_chrom 12

## 2.train model
./ce_train.py -o test_train_out -l test_log_out params.yaml 23tissues --seed 42

# (optional) or try one experiment with timestamp 
time=$(date +"%Y%m%d_%H%M%S")
nohup ./ce_train.py  -o train_out/$time -l tensorboard/$time \
    params.yaml 23tissues > logs/$time.log &

# (optional) hyperparameters experiments
./ce_train_grid.py -p 1 -t 23tissues -o experiments/grid_search_20250526_night \
    params_grid.yaml -r -s 1 100 200 300 400

# (optional) training visualization
tensorboard --logdir=./test_log_out --host 0.0.0.0 --port 6006

## 3. evaluate model in test sets
./ce_eval.py -o test_eval_out --rank --save -m --shifts 0 --step 1 \
    -t 23tissues/targets.txt --split test params.yaml \
    test_train_out/model_best.h5 23tissues