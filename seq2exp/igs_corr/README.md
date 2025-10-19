## example
```sh
prefix=P8Exp2100_129tracks_34P
seed=100
model_path=../tf_exps/${seed}_${prefix}_train_transfer_out/model_best.h5
yaml=../tf_exps/${seed}_${prefix}_train_transfer_out/params.yaml 
paddy_igs_region.py -o ${prefix}_${seed}.igs_region.h5 --num_steps 100 \
    --model_path $model_path --regions_csv PBT_starrseq_rp1_0based.csv \
    $yaml NumChr.Rice_MSUv7.fa Rice_MSUv7.gff3 
```

## Reproduce
To reduce runtime, split into chr-level csv and distribute to multiple GPUs to run.
```sh
input="PBT_starrseq_rp1_0based.csv"

PREFIX=$(basename "$input" .csv)
mkdir -p "$PREFIX"

awk -F',' -v prefix="$PREFIX" '
NR==1 {header=$0; next}
{
    outfile = prefix "/chr" $1 ".csv"
    if (!(outfile in seen)) {
        print header > outfile
        seen[outfile] = 1
    }
    print $0 >> outfile
}' "$input"

```
manually distribute into multiple GPUs
```sh
prefix=P8Exp2100_129tracks_34P
seed=100
model_path=../tf_exps/${seed}_${prefix}_train_transfer_out/model_best.h5
yaml=../tf_exps/${seed}_${prefix}_train_transfer_out/params.yaml 

i=1 # 1...12
csv=$PREFIX/chr${i}.csv
CUDA_VISIBLE_DEVICES=1 \
    nohup paddy_igs_region.py -o $PREFIX/${prefix}_${seed}.chr${i}.igs_region.h5 --num_steps 50 \
        --model_path $model_path --regions_csv $csv \
        $yaml NumChr.Rice_MSUv7.fa Rice_MSUv7.gff3 > $PREFIX/chr${i}_igs_region.log 2>&1 &
    
```