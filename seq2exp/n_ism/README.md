```sh
prefix=P8Exp2100_129tracks_34P
seed=100
model_path=../tf_exps/${seed}_${prefix}_train_transfer_out/model_best.h5
yaml=../tf_exps/${seed}_${prefix}_train_transfer_out/params.yaml
```
## Example

```sh
paddy_Nism_gene.py --model_path $model_path $yaml NumChr.Rice_MSUv7.fa test.gff3
```

## Parallel
```sh
mkdir -p sub_gffs && cd sub_gffs

total=$(awk '$3=="gene"{c++}END{print c}' ../Rice_MSUv7.chr.gff3)
per=$(( (total + 15) / 16 ))
awk -v per=$per '
$3=="gene"{count++}
$3=="gene" && count % per == 1 {file=sprintf("P8_part_%02d.gff", int((count-1)/per))}
{if(file) print > file}
' ../Rice_MSUv7.chr.gff3

cd ..

PREFIX=${prefix}_${seed}

mkdir -p $PREFIX
conda activate paddy

IDX_LIST=$(seq 0 15) # (16 31)
for idx in $IDX_LIST; do
    gpu_id=$(( idx / 2 ))
    CUDA_VISIBLE_DEVICES=$gpu_id nohup paddy_Nism_gene.py -o $PREFIX/${PREFIX}_${idx}_Nism_gene.h5 --model_path $model_path $yaml NumChr.Rice_MSUv7.fa sub_gffs/P8_part_$(printf "%02d" $idx).gff > $PREFIX/${idx}_Nism_gene.log 2>&1 &
done

IDX_LIST=$(seq 0 15)
for idx in $IDX_LIST; do
    gpu_id=$(( idx / 2 ))
    CUDA_VISIBLE_DEVICES=$gpu_id nohup paddy_Nism_gene.py -o $PREFIX/${PREFIX}_$((idx + 16))_Nism_gene.h5 --model_path $model_path $yaml NumChr.Rice_MSUv7.fa sub_gffs/P8_part_$(printf "%02d" $((idx + 16))).gff > $PREFIX/$((idx + 16))_Nism_gene.log 2>&1 &
done
```

