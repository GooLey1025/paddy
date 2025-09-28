```shell
paddy_embed.py transfer_embed.yaml ../seq2exp/100_P8_106tracks_34P_train_transfer_out/model_best.h5 -f G.fa --layer_index 89 --output_name G_embedding.h5 --auto_analyze
paddy_embed.py  --compare embed_out/A_embedding.h5 embed_out/G_embedding.h5
```