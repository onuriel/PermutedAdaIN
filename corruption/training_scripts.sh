#ResNeXt:
python cifar.py -m resnext -e 200 --padain 0.01 -s resnext_padain

# DenseNet:
python cifar.py -m densenet -e 200 -wd 0.0001 --padain 0.01 -s densenet_padain