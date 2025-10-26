```sh
prefix=P8Exp2100_129tracks_34P
seed=100
model_path=../tf_exps/${seed}_${prefix}_train_transfer_out/model_best.h5
yaml=../tf_exps/${seed}_${prefix}_train_transfer_out/params.yaml
```
## Example

```sh
paddy_ism_gene.py --model_path $model_path $yaml NumChr.Rice_MSUv7.fa test.gff3
```

## Parallel
```sh
mkdir -p sub_gffs && cd sub_gffs
awk '$3 == "gene" {count++} $3 == "gene" && count % 1800 == 1 {file="P8_part_" sprintf("%02d", int((count-1)/1800)) ".gff"} {if(file) print > file}' ../Rice_MSUv7.gff3
cd ..

PREFIX=${prefix}_${seed}

mkdir -p $PREFIX
conda activate paddy

IDX_LIST=$(seq 0 15) # (16 31)
for idx in $IDX_LIST; do
    gpu_id=$(( idx / 2 ))
    CUDA_VISIBLE_DEVICES=$gpu_id nohup paddy_ism_gene.py -o $PREFIX/${PREFIX}_${idx}_ism_gene.h5 --num_steps 50 --bath_size 4 --model_path $model_path $yaml NumChr.Rice_MSUv7.fa sub_gffs/P8_part_$(printf "%02d" $idx).gff > $PREFIX/${idx}_ism_gene.log 2>&1 &
done

IDX_LIST=$(seq 0 15)
for idx in $IDX_LIST; do
    gpu_id=$(( idx / 2 ))
    CUDA_VISIBLE_DEVICES=$gpu_id nohup paddy_ism_gene.py -o $PREFIX/${PREFIX}_$((idx + 16))_ism_gene.h5 --num_steps 50 --model_path $model_path $yaml NumChr.Rice_MSUv7.fa sub_gffs/P8_part_$(printf "%02d" $((idx + 16))).gff > $PREFIX/$((idx + 16))_ism_gene.log 2>&1 &
done
```

