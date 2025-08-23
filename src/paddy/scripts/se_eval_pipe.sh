#!/bin/bash

# #########################################################
# se_eval_pipe.sh
# ##########################################################
# evaluate the trained model on a set of sequences from different rice genomes.

# E.g: 
# se_eval_pipe.sh transfer_CE_PaddyHead.yaml \
# 129tracks_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5 \
# new_ATG_UpDown16K_transfer_data_out
# new_ATG_UpDown16K_transfer_data_out/unique_identifiers.txt
# se_eval_exps
params=$1
model=$2
data_dir=$3
identifier_file=$4
output_dir=$5
flag=0
mkdir -p $output_dir
> $output_dir/all_metrics.tsv

cat $identifier_file | while read identifier; do
    echo "Evaluating $identifier"
    se_eval.py $params $model $data_dir --tfr_pattern test-$identifier*.tfr -o $output_dir/$identifier \
        || echo "Error: se_eval.py $params $model $data_dir --tfr_pattern test-$identifier*.tfr -o $output_dir/$identifier" 
    # merge metrics.txt into a single file
    
    # if wrong, exit
    if [ $? -ne 0 ]; then
        echo "Error: se_eval.py failed" 
        exit 1
    fi

    if [ $flag = 0 ]; then
        # add header
        cat $output_dir/$identifier/metrics.tsv >> $output_dir/all_metrics.tsv
        flag=1
    else
        awk 'NR==2' $output_dir/$identifier/metrics.tsv >> $output_dir/all_metrics.tsv
    fi
    echo "Done $identifier"
done



# replace test-P10_9311*.tfr by P10_9311
sed -i 's/test-//g; s/\*\.tfr//g' $output_dir/all_metrics.tsv

# insert a column "used_model" at the first column, and fill it with $model

if [[ $model == *.h5 ]]; then
    model_no_ext=${model%.h5}
else
    model_no_ext=$model
fi

awk -v m="$model_no_ext" 'NR==1{print "used_model\t"$0; next} {print m"\t"$0}' "$output_dir/all_metrics.tsv" > tmp
mv tmp "$output_dir/all_metrics.tsv"

echo "se_eval pipe done. All metrics are saved in $output_dir/all_metrics.tsv"



