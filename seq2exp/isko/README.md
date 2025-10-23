```sh
prefix=PBT_starrseq_rp1
awk -F, 'NR>=2 {print $1"\t"($2-1)"\t"$3}' $prefix.csv | sed 's/chr//g' > $prefix.bed
paddy_n_ism.py --ref ../NumChr.Rice_MSUv7.fa --gtf Rice_MSUv7.gff3  --stats SED,logSED,D1,logD1,D2,logD2,nD2,JS --model_path ../tf_exps/100_P8Exp2100_129tracks_34P_train_transfer_out/model_best.h5 -o ${prefix}_n_sim_out  ../tf_exps/100_P8Exp2100_129tracks_34P_train_transfer_out/params.yaml ${prefix}.bed

awk -F, 'NR==1 {print} NR>=2 {gsub(/chr/, "", $1); $2=($2-1); print}' OFS=, ${prefix}.csv > ${prefix}_0based.csv
python corr_compare.py  --metrics SED logSED D1 logD1 D2 logD2 nD2 JS --h5 ${prefix}_n_sim_out/ism.h5 --csv ${prefix}_0based.csv -o ${prefix}_plots
```
```sh

borzoi_sad.py -f NumChr.Rice_MSUv7_chrHead.fa -o rp1_sad --stats SAD,SADlog,logSAD,sqrtSAD,SAX,D1,logD1,logD2,D2,sqrtD2,JS,logJS,REF,ALT -t 129targets_formatted.txt ../129_pretrain_model_ablations/exp2_100/params.json ../129_pretrain_model_ablations/exp2_100/model_best.h5 PBT_starrseq_rp1.vcf

python3 sad_corr_compare.py --h5 rp1_sad/sad.h5 -o 129targets_rp1_sad_out --csv PBT_starrseq_rp1_0based.csv 
```
