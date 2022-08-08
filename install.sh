#!/usr/bin/env bash

# install requirements
pip install torch==1.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
pip install lmdb
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-geometric==1.7.2
pip install tensorboardX==2.4.1
pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
pip install performer-pytorch
pip install tensorboard
pip install setuptools==59.5.0

# install submodules
git submodule update --init --recursive
(
cd large-scale-regression/fairseq || exit
pip install .
python setup.py build_ext --inplace
cd ../.. || exit
)

# prevent train hang
pip uninstall setuptools -y
