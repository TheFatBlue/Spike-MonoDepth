export LD_PRELOAD=/root/miniconda3/pkgs/libstdcxx-ng-11.2.0-h1234567_1/lib/libstdc++.so.6.0.29:$LD_PRELOAD

# rm -rf s2d_checkpoints

python='/root/miniconda3/envs/scv/bin/python'

model_path='ft_il_100e_cat_4cdr'

python test_DENSE.py \
    --path_to_model /root/autodl-tmp/$model_path/train_s2d_SpikeTransformer/checkpoint-epoch010-loss-0.1014.pth.tar \
    --output_path /root/autodl-tmp/$model_path \
    --data_folder /root/autodl-tmp/Spike-Stero/test \
    --config configs/$model_path.json
