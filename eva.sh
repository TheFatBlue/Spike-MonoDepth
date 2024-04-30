export LD_PRELOAD=/root/miniconda3/pkgs/libstdcxx-ng-11.2.0-h1234567_1/lib/libstdc++.so.6.0.29:$LD_PRELOAD

# rm -rf s2d_checkpoints

python='/root/miniconda3/envs/scv/bin/python'
output_folder='/root/autodl-tmp/baseline'

python evaluation_DENSE.py \
    --target_dataset $output_folder/ground_truth/npy/depth_image/ \
    --predictions_dataset $output_folder/npy/image/ \
    --clip_distance 4.0 \
    --reg_factor 1.86 \
    --output_folder /root/autodl-tmp/baseline_eva \
    --debug