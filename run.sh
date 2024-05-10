export LD_PRELOAD=/root/miniconda3/pkgs/libstdcxx-ng-11.2.0-h1234567_1/lib/libstdc++.so.6.0.29:$LD_PRELOAD

# rm -rf s2d_checkpoints

python='/root/miniconda3/envs/scv/bin/python'

python train.py --config /root/code/smde/configs/ft_il_100e.json \
                --datafolder /root/autodl-tmp \
                --initial_checkpoint /root/autodl-tmp/train_ib_150e_cat_4cdr/train_s2d_SpikeTransformer/checkpoint-epoch150-loss-0.0266.pth.tar
                # ; /usr/bin/shutdown
