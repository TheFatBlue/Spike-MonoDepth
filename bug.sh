pip uninstall setuptools
pip install setuptools==59.5.0

pip install torch==1.10.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/cu110/torch_stable.html

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge