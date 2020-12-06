#train resnet50 on imagenet with padain distributed for 300 epochs

python imagenet.py --use-dataset-path <PATH_TO_IMAGENET_DATASET>   -a resnet50  --exp-name resnet50_imagenet_padain_0.01  --padain 0.01  --dist-url 'tcp://127.0.0.1:8994'  --dist-backend 'nccl'  --multiprocessing-distributed  --world-size 1 --rank 0  -p 100  -j 10  --epochs 300  --wd 0.0001

#train resnet18 on cifar100

python cifar100.py -net resnet18 -gpu 0 -padain 0.01