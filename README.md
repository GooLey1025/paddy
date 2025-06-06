## Examples
```bash
git clone https://github.com/GooLey1025/paddy.git
echo "export PATH=\$PATH:$(pwd)/paddy/src/paddy/scripts" >> ~/.bashrc
source ~/.bashrc
conda create -n paddy python=3.10
conda activate paddy
cd paddy
pip install -e .
cd examples

## 1.prepare data
# generate tfrecords via random split train_valid_test sets
paddy_data.py --h5_dir data/seqs_cov --output_dir 23tissues \
    --xlsx_file Nip_ATGsite_UD16K_Bed_new.xlsx \
    --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1
# or generate tfrecords via specify a specific chrom as the test or valid set, the rest are train set.
paddy_data.py --h5_dir data/seqs_cov --output_dir 23tissues \
    --xlsx_file Nip_ATGsite_UD16K_Bed_new.xlsx \
    --valid_chrom 11 --test_chrom 12

## 2.train model
paddy_train.py -m -o test_train_out -l test_log_out params.yaml 23tissues

# (optional) or try one experiment with timestamp 
time=$(date +"%Y%m%d_%H%M%S")
nohup paddy_train.py  -o train_out/$time -l tensorboard/$time \
    params.yaml 23tissues > logs/$time.log &

# (optional) hyperparameters experiments
paddy_train_grid.py -p 1 -t 23tissues -o experiments/grid_search_20250526_night \
    params_grid.yaml -r

# (optional) training visualization
tensorboard --logdir=./test_log_out --host 0.0.0.0 --port 6006

## 3. evaluate model in test sets
paddy_eval.py -o test_eval_out --rank --save -m --shifts 0 --step 1 \
    -t 23tissues/targets.txt --split test params.yaml \
    test_train_out/model_best.h5 23tissues
