export LD_PRELOAD=/root/miniconda3/pkgs/libstdcxx-ng-11.2.0-h1234567_1/lib/libstdc++.so.6.0.29:$LD_PRELOAD

# rm -rf s2d_checkpoints

python='/root/miniconda3/envs/scv/bin/python'

python test_model.py --config /root/code/smde/configs/ft_ol_60e.json \
                --data_folder /root/code/smde/dataset/Spike-Stero/test \
                --path_to_model /root/autodl-tmp/SpikeT/model_best.pth.tar