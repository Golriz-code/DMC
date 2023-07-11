#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"



#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.



#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.



#SBATCH --time=0-03:00
source /home/golriz/projects/def-guibault/golriz/env/bin/activate
HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install 

# NOTE: For GRNet 

# Cubic Feature Sampling
cd $HOME/extensions/cubic_feature_sampling
python setup.py install 

# Gridding & Gridding Reverse
cd $HOME/extensions/gridding
python setup.py install 

# Gridding Loss
cd $HOME/extensions/gridding_loss
python setup.py install 

