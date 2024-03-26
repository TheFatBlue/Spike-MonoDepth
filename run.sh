export LD_PRELOAD=/root/miniconda3/pkgs/libstdcxx-ng-11.2.0-h1234567_1/lib/libstdc++.so.6.0.29:$LD_PRELOAD

# rm -rf s2d_checkpoints

python='/root/miniconda3/envs/scv/bin/python'

python train.py --config /root/code/smde/configs/train_s2d_spikeT.json \
                --datafolder /root/autodl-tmp \
                --multiprocessing_distributed