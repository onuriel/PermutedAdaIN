This is code provided as supplementary for the paper : Permuted AdaIN: Reducing the Bias Towards Global Statistics in Image Classification

##Sections

# Image Classification Section 4.1 in paper
 - ImageNet + Cifar100 experiments - under classification folder see classification/training_script.sh for examples of running experiments
 - ImageNet code is based on official pytorch ImageNet training script: https://github.com/pytorch/examples/tree/master/imagenet
 - CIFAR100 code is based on git https://github.com/weiaicunzai/pytorch-cifar100

# Domain Adaption Section 4.3 in paper
 - Code based on official FADA implementation : https://github.com/JDAI-CV/FADA
 - An examples of running the three stage experiment sequentially is given in FADA/train_with_sd.sh

# Robustness Towards Corruption Section 4.5 in paper
 - Code based on https://github.com/google-research/augmix
 - Example of running the experiment is given in corruption/training_scripts.sh

