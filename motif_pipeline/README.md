## Pipeline
```sh
grads_dir=5tissues_atg_grads
TISSUE_INDICES="0,1,14,18,20" # This 
select_top_gene.py P8_ATG_UD16K.exp --tissue_indices $TISSUE_INDICES -o diff_expr
# extract GFF files for specific tissues
make_top_genes_gff.py --tissue_indices "$TISSUE_INDICES" \
    --data_dir diff_expr \
    --gtf Rice_MSUv7.gff3 \
    -o selected_genes_gff \
    --gene_id_column "gene_id"

mkdir -p $grads_dir
IFS=',' read -ra INDICES <<< $TISSUE_INDICES
for i in ${INDICES[@]}; do
    paddy_grad_gene.py \
    --model_path ../seq2exp/best_model_dirs/34P_23tracks_34P_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5 \
    --fa NumChr.Rice_MSUv7.fa --rc --atg \
    ../seq2exp/transfer_CE_PaddyHead.yaml \
    -o $grads_dir/tissue_${i}.h5 \
    selected_genes_gff/tissue_${i}_top_genes.gff3
done
# --rc --untransform_old --track_scale 0.01 --track_transform 0.75 --clip_soft 384.0

# select specify
mkdir -p modisco_${grads_dir}_preprocessed 
preprocess_for_modisco.py \
    --grad_dir $grads_dir \
    --tissue_indices $TISSUE_INDICES \
    -o modisco_${grads_dir}_preprocessed \
    --residual \
    --gaussian_sigma 1280 \
    --gaussian_truncate 2.0

modisco motifs \
        -i modisco_${grads_dir}_preprocessed/modisco_preprocessed_all.h5 \
        -o modisco_${grads_dir}_results/modisco_tissues.h5 \
        -n 40000 -t 24 -g 8 -z 18 -f 8 -w 32768 -v 

modisco meme -i modisco_${grads_dir}_results/modisco_tissues.h5 \
    -t PFM -o modisco_${grads_dir}_results/tissues_motifs.meme
trim_meme_ic.py modisco_${grads_dir}_results/tissues_motifs.meme \
    modisco_${grads_dir}_results/tissues_motifs_ic_trimmed.meme \
    --ic-thresh 0.1   

mkdir -p tomtom_${grads_dir}_results
tomtom -oc tomtom_${grads_dir}_results/tissues -evalue \
    modisco_${grads_dir}_results/tissues_motifs_ic_trimmed.meme \
    JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt

modisco_analysis.py 
```

## Old version
```sh 
mkdir -p modisco_${grads_dir}_results
for i in $(echo $TISSUE_INDICES | tr ',' ' '); do
    modisco motifs \
        -i modisco_${grads_dir}_preprocessed/tissue_${i}_preprocessed.h5 \
        -o modisco_${grads_dir}_results/modisco_tissue_${i}.h5 \
        -n 40000 -t 24 -g 8 -z 18 -f 8 -v
done

# caculate GC_content
# GC= $(bioawk -c fastx '{n=gsub(/[GCgc]/,"&",$seq); m=length($seq)-gsub(/[^ACGTacgt]/,"&",$seq); if(m==0) f=0; else f=n/m; print $name, m, n, f}' NumChr.Rice_MSUv7.fa \
#| awk '{sum+=$4} END {print sum/NR}' )

IFS=',' read -ra INDICES <<< "$TISSUE_INDICES"
for tissue in "${INDICES[@]}"; do
    modisco meme -i modisco_${grads_dir}_results/modisco_tissue_${tissue}.h5 \
                 -t PFM \
                 -o modisco_${grads_dir}_results/tissue_${tissue}_motifs.meme
    trim_meme_ic.py modisco_${grads_dir}_results/tissue_${tissue}_motifs.meme \
        modisco_${grads_dir}_results/tissue_${tissue}_motifs_ic_trimed.meme \
        --ic-thresh 0.1
done
```
```sh

mkdir -p tomtom_${grads_dir}_results
IFS=',' read -ra INDICES <<< "$TISSUE_INDICES"
for tissue in "${INDICES[@]}"; do
    tomtom \
    -oc tomtom_${grads_dir}_results/tissue_${tissue} \
    -evalue \
    modisco_${grads_dir}_results/tissue_${tissue}_motifs_ic_trimed.meme \
    JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt
done





# Extract motif ID and name from JASPAR MEME file to create mapping file
grep "MOTIF" JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt | awk '{print $2"\t"$3}' > JASPAR_mapping.tsv

IFS=',' read -ra INDICES <<< "$TISSUE_INDICES"
for i in "${INDICES[@]}"; do

    # Select best hit (lowest p-value) for each Query_ID from tomtom results
    #   1) Keep header
    #   2) Sort by Query_ID then p-value
    #   3) Keep unique Query_ID (best hit)
    #   4) Remove comment lines (#) and empty lines
    (head -n 1 tomtom_${grads_dir}_results/tissue_${i}/tomtom.tsv && tail -n +2 tomtom_${grads_dir}_results/tissue_${i}/tomtom.tsv | sort -k1,1 -k4,4g | sort -u -k1,1) | grep -v "#" | grep -v "^$" > tomtom_${grads_dir}_results/tissue_${i}_tmp.tsv

    # Map Target_ID to JASPAR motif name
    awk 'BEGIN{FS=OFS="\t"}
     NR==FNR{
         # build mapping: same ID may have multiple names -> join with commas
         map[$1] = (map[$1] ? map[$1] "," $2 : $2);
         next
     }
     FNR==1{
         # print header with inserted column name after col2
         printf "%s%s%s%s%s", $1, OFS, $2, OFS, "Motif_JASPAR";
         for(i=3;i<=NF;i++) printf "%s%s", OFS, $i;
         printf "\n";
         next
     }
     {
         # data lines: print col1, col2, mapped name (or NA), then cols 3..NF
         printf "%s%s%s%s%s", $1, OFS, $2, OFS, (map[$2] ? map[$2] : "NA");
         for(i=3;i<=NF;i++) printf "%s%s", OFS, $i;
         printf "\n";
     }' JASPAR_mapping.tsv tomtom_${grads_dir}_results/tissue_${i}_tmp.tsv tomtom_${grads_dir}_results/tissue_${i}_bestHit_motif.tsv
done

motif_metadata_summary.py \
    --modisco_dir modisco_${grads_dir}_results \
    --tomtom_dir tomtom_${grads_dir}_results \
    --tissue_indices $TISSUE_INDICES \
    --jaspar_meme JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt \
    --grads_dir ${grads_dir} \
    --output integrated_motifs_${grads_dir}.h5 \
    --verbose

paddy_motif_plot.py -i integrated_motifs_${grads_dir}.h5 -o plots -f pdf -g ${grads_dir}

```