export LD_PRELOAD=/root/miniconda3/pkgs/libstdcxx-ng-11.2.0-h1234567_1/lib/libstdc++.so.6.0.29:$LD_PRELOAD

# rm -rf s2d_checkpoints

python='/root/miniconda3/envs/scv/bin/python'

python test_DENSE.py \
    --path_to_model /root/autodl-tmp/ft_il_100e/train_s2d_SpikeTransformer/model_best.pth.tar \
    --output_path /root/autodl-tmp/ft_il_100e \
    --data_folder /root/autodl-tmp/Spike-Stero/test \
    --config configs/ft_il_100e.json