```sh
seed=100
prefix=exp1_${seed}
mkdir -p $prefix
cp 129targets.json $prefix/$prefix.json
#cp exp1_100/exp1_100.json $prefix/$prefix.json 
# manually edit json to do experiments
CUDA_VISIBLE_DEVICES=6,7 nohup hound_train_seed.py -s $seed -o $prefix/$prefix_train_out -l $prefix/log_dir $prefix/$prefix.json ../129_mseqs > $prefix/$prefix.train.log &
```
