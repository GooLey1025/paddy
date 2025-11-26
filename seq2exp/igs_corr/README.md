## Example

```sh
prefix=P8Exp2100_129tracks_34P
seed=100
model_path=../tf_exps/${seed}_${prefix}_train_transfer_out/model_best.h5
yaml=../tf_exps/${seed}_${prefix}_train_transfer_out/params.yaml 

paddy_igs_gene.py --model_path $model_path $yaml NumChr.Rice_MSUv7.fa test.gff3 -o test_gene_outh5   
```

## Distribute into multiple GPUs
```sh
mkdir -p sub_gffs && cd sub_gffs
total=$(awk '$3=="gene"{c++}END{print c}' ../Rice_MSUv7.chr.gff3)
per=$(( (total + 11) / 12 ))
awk -v per=$per '
$3=="gene"{count++}
$3=="gene" && count % per == 1 {file=sprintf("P8_part_%02d.gff", int((count-1)/per))}
{if(file) print > file}
' ../Rice_MSUv7.chr.gff3
cd ..

#awk '$3 == "gene" {count++} $3 == "gene" && count % 1800 == 1 {file="P8_part_" sprintf("%02d", int((count-1)/1800)) ".gff"} {if(file) print > file}' ../Rice_MSUv7.gff3
```

manually distribute into multiple GPUs

```sh
prefix=P8Exp2100_129tracks_34P
seed=100
model_path=../tf_exps/${seed}_${prefix}_train_transfer_out/model_best.h5
yaml=../tf_exps/${seed}_${prefix}_train_transfer_out/params.yaml 

PREFIX=${prefix}_${seed}

mkdir -p $PREFIX
conda activate paddy

id=00
gpu_id=6
CUDA_VISIBLE_DEVICES=$gpu_id nohup paddy_igs_gene.py --model_path $model_path $yaml NumChr.Rice_MSUv7.fa sub_gffs/P8_part_${id}.gff -o $PREFIX/${PREFIX}_${id}_ig_gene.h5 > $PREFIX/${id}_ig_gene.log 2>&1 &

IDX_LIST="0 1 2 3 4 5 12 13 14 15"
IDX_LIST=$(seq 6 11) #0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
for idx in $IDX_LIST; do
    gpu_id=$(( idx / 2 ))
    CUDA_VISIBLE_DEVICES=$gpu_id nohup paddy_igs_gene.py -o $PREFIX/${PREFIX}_${idx}_ig_gene.h5 --num_steps 50 --model_path $model_path $yaml NumChr.Rice_MSUv7.fa sub_gffs/P8_part_$(printf "%02d" $idx).gff > $PREFIX/${idx}_ig_gene.log 2>&1 &
done

IDX_LIST=$(seq 0 15)
for idx in $IDX_LIST; do
    gpu_id=$(( idx / 2 ))
    CUDA_VISIBLE_DEVICES=$gpu_id nohup paddy_igs_gene.py -o $PREFIX/${PREFIX}_$((idx + 15))_ig_gene.h5 --num_steps 50 --model_path $model_path $yaml NumChr.Rice_MSUv7.fa sub_gffs/P8_part_$(printf "%02d" $((idx + 15))).gff > $PREFIX/$((idx + 15))_ig_gene.log 2>&1 &
done

```



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
CUDA_VISIBLE_DEVICES=1 nohup paddy_igs_region.py -o $PREFIX/${prefix}_${seed}.chr${i}.igs_region.h5 --num_steps 50 --model_path $model_path --regions_csv $csv $yaml NumChr.Rice_MSUv7.fa Rice_MSUv7.gff3 > $PREFIX/chr${i}_igs_region.log 2>&1 &
    
```
