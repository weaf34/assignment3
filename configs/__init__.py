import os
maindir = os.path.realpath('.')

"""
DATA PATH
"""
train_path = f'{maindir}/data/hotdog/train/'
test_path = f'{maindir}/data/hotdog/test/'
train_path101 = f'{maindir}/data/food101small/train/'
test_path101 = f'{maindir}/data/food101small/test/'

epoch = 25
model_path = f'{maindir}/archive/best.pth'
metrics_path= f'{maindir}/archive/metrics.txt'

model_path101 = f'{maindir}/archive/best101.pth'
metrics_path101 = f'{maindir}/archive/metrics101.txt'

lr = 0.0001

# Note this can be done programmatically in more smarter way
label_101_dict = {name: i for i, name in enumerate(os.listdir(test_path101))}

