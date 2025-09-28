```sh
cka_repr_models_compute.py -m1p 23ave.params_micro_106targets.yaml -m1w ../seq2exp/SC_models/34P_Borzoi_Seq2RNAseqTracks_rp1_model_best.h5 -m2p 23ave.params_micro_106targets.yaml -m2w ../seq2exp/SC_models/34P_Borzoi_Seq2RNAseqTracks_rp2_model_best.h5  -d ../seq2exp/23_P8_mseqs -o 34P_23_SCmodel_cka_output -ms 512 -po

nohup cka_repr_models_compute.py -m1p ../seq2exp/transfer_CE_PaddyHead.yaml \
    -m1w ../seq2exp/best_model_dirs/34Prp1_23tracks_34P_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5 \
    -m2p ../seq2exp/transfer_CE_PaddyHead.yaml \
    -m2w ../seq2exp/best_model_dirs/34Prp2_23tracks_34P_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5  \
    -d ../seq2exp/P8_new_ATG_UpDown16K_transfer_data_out -o 34P_23_34P_rp1rp2_cka_output -ms 64 > log &

nohup cka_repr_models_compute.py -m1p P8_106_SC.yaml \
    -m1w ../seq2exp/SC_models/P8_106_Pretrain1.model_best.h5 \
    -m2p P8_106_SC.yaml \
    -m2w ../seq2exp/SC_models/P8_106_Pretrain2.model_best.h5 \
    -d ../seq2exp/23_P8_mseqs -o P8_106_Pretrain_rp1_rp2_cka_output -ms 64 > log &
    
    cka_repr_models_compute.py -m1p ../seq2exp/transfer_CE_PaddyHead.yaml \
    -m1w ../seq2exp/best_model_dirs/34Prp1_23tracks_34P_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5 \
    -m2p ../seq2exp/transfer_CE_PaddyHead.yaml \
    -m2w ../seq2exp/best_model_dirs/34Prp1_23tracks_34P_TrunkFrozen_PaddyHead_best_model_dir/seed200_model_best.h5  \
    -d ../seq2exp/P8_new_ATG_UpDown16K_transfer_data_out -o 34Prp1_23_34P_seed100_200_cka_output -ms 64
```