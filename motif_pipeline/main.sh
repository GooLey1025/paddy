#!/bin/bash

# Configuration
export grads_dir=106rp3seed100_5tissues_atg_fw_grads
export TISSUE_INDICES="0,1,14,18,20"

# Windowing parameters
export up_size=2000 # comment this out if you don't want to use the output window
export down_size=0   # set downstream size (0 for upstream-only analysis)

# Set window size and directory naming based on whether windowing is used
if [ -z "$up_size" ]; then
    export WINDOW=32768
    export window_grads_dir=${WINDOW}_${grads_dir}
    echo "Running pipeline without windowing (full center_size=32768)"
else
    # only care up_size
    export WINDOW=$up_size
    export window_grads_dir=up_${up_size}_down_${down_size}_${grads_dir}
    echo "Running pipeline with windowing: up_size=${up_size}, down_size=${down_size}, total_window=${WINDOW}"
fi

echo "Output directory suffix: ${window_grads_dir}"
echo "Processing tissues: ${TISSUE_INDICES}"
echo "=================================================================================="

# If you have runed the following code, you can skip this step
mkdir -p $grads_dir
IFS=',' read -ra INDICES <<< $TISSUE_INDICES
for i in ${INDICES[@]}; do
    paddy_grad_fa.py \
    --model_path ../seq2exp/best_model_dirs/P8rp3_106tracks_34P_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5 \
    ../seq2exp/transfer_CE_PaddyHead.yaml \
    -o $grads_dir/tissue_${i}.h5 \
    diff_expr/tissue_${i}_top_genes.fa
done

# Step 1: Preprocess gradients for modisco
echo "Step 1: Preprocessing gradients..."
if [ -z "$up_size" ]; then
    echo "Using full center_size without windowing"
    preprocess_for_modisco.py \
        --grad_dir $grads_dir \
        --tissue_indices $TISSUE_INDICES \
        -o ${grads_dir} \
        --out_format h5 \
        --center_size 32768 \
        --residual \
        --gaussian_sigma 1280 \
        --gaussian_truncate 2.0 \
        --split_by_tissues
else 
    echo "Using windowed preprocessing: up_size=${up_size}, down_size=${down_size}"
    preprocess_for_modisco.py \
        --grad_dir $grads_dir \
        --tissue_indices $TISSUE_INDICES \
        -o ${grads_dir} \
        --out_format h5 \
        --center_size 32768 \
        --residual \
        --gaussian_sigma 1280 \
        --gaussian_truncate 2.0 \
        --split_by_tissues \
        --up_size $up_size \
        --down_size $down_size
fi
echo "Preprocessing completed."
echo "=================================================================================="



# Step 2: Run modisco motif discovery
echo "Step 2: Running modisco motif discovery..."
echo "Window size for modisco: ${WINDOW}"
mkdir -p modiscolite_${window_grads_dir}_results
parallel -j 5 --bar --halt now,fail=1 \
    modisco motifs \
        -i ${grads_dir}/modisco_preprocessed_tissue_{}.h5 \
        -o modiscolite_${window_grads_dir}_results/modiscolite_tissue_{}.h5 \
        -w $WINDOW -n 40000 -t 24 -g 8 -z 18 -f 8 -v \
    ::: ${TISSUE_INDICES//,/ }
echo "Modisco motif discovery completed."
echo "=================================================================================="

# Step 3: Convert motifs to MEME format and trim by information content
echo "Step 3: Converting motifs to MEME format and trimming..."
IFS=',' read -ra INDICES <<< "$TISSUE_INDICES"
for tissue in "${INDICES[@]}"; do
    echo "Processing tissue ${tissue}..."
    modisco meme -i modiscolite_${window_grads_dir}_results/modiscolite_tissue_${tissue}.h5 \
                 -t PFM \
                 -o modiscolite_${window_grads_dir}_results/tissue_${tissue}_motifs.meme
    trim_meme_ic.py modiscolite_${window_grads_dir}_results/tissue_${tissue}_motifs.meme \
        modiscolite_${window_grads_dir}_results/tissue_${tissue}_motifs_ic_trimed.meme \
        --ic-thresh 0.1
done
echo "MEME format conversion and trimming completed."
echo "=================================================================================="

# Step 4: Run Tomtom motif comparison against JASPAR database
echo "Step 4: Running Tomtom motif comparison..."
parallel -j 5 --halt now,fail=1 '
    echo "Running Tomtom for tissue {}"
    tomtom \
    -oc modiscolite_${window_grads_dir}_results/tomtom_tissue_{} \
    -evalue \
    modiscolite_${window_grads_dir}_results/tissue_{}_motifs_ic_trimed.meme \
    JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt
' ::: ${TISSUE_INDICES//,/ }
echo "Tomtom comparison completed."
echo "=================================================================================="

# Step 5: Run motif analysis with coordinate conversion
echo "Step 5: Running motif analysis..."
if [ -z "$up_size" ]; then
    echo "Running motif analysis without windowing parameters"
    motif_analysis.py -t $TISSUE_INDICES --data_dir modiscolite_${window_grads_dir}_results \
        --grads_dir $grads_dir --output_dir motif_anaylsis_${window_grads_dir}_results \
        --window_size 32768
else
    echo "Running motif analysis with windowing parameters: up_size=${up_size}, down_size=${down_size}"
    motif_analysis.py -t $TISSUE_INDICES --data_dir modiscolite_${window_grads_dir}_results \
        --grads_dir $grads_dir --output_dir motif_anaylsis_${window_grads_dir}_results \
        --window_size 32768 --up_size $up_size --down_size $down_size
fi
echo "Motif analysis completed."
echo "=================================================================================="

# Step 6: Generate motif plots
echo "Step 6: Generating motif plots..."
motif_plot.py --h5_file motif_anaylsis_${window_grads_dir}_results/motif_clusters.h5 \
    --output_dir motif_plots_${window_grads_dir} --tissue_dict "23tissues_dict.json" \
    --n_process 32 --min_limit -0.015 --max_limit 0.015 \
    --tissue_colors "#C10606,#C2560F,#196E9B,#508A65,#2A386F" --image_format pdf
echo "Motif plotting completed."
echo "=================================================================================="

echo "Pipeline completed successfully!"
echo "Results directories:"
echo "- Modisco results: modiscolite_${window_grads_dir}_results/"
echo "- Motif analysis: motif_anaylsis_${window_grads_dir}_results/"
echo "- Plots: motif_plots_${window_grads_dir}/"
if [ ! -z "$up_size" ]; then
    echo ""
    echo "Windowing was used: up_size=${up_size}, down_size=${down_size}"
    echo "Coordinate conversion was applied in motif analysis step."
fi
