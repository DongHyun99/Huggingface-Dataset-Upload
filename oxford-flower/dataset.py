import os
import shutil

from datasets import load_dataset, Dataset, ClassLabel
from huggingface_hub import notebook_login

data = open('label.txt', 'r')
train = open('train.txt', 'r')
test = open('test.txt', 'r')
validation = open('validation.txt', 'r')
label_name = open('label_name.txt', 'r')
imgDir = 'jpg'

labels = data.read()
train_labels = train.read()
test_labels = test.read()
validation_labels = validation.read()
label_names = label_name.read()
imgList = os.listdir(imgDir)

labels = [int(label) for label in labels.split(',')]
label_names = [label for label in label_names.split(',')]
train_labels = [int(label) for label in train_labels.split(',')]
test_labels = [int(label) for label in test_labels.split(',')]
validation_labels = [int(label) for label in validation_labels.split(',')]

os.makedirs('oxford-flower/train', exist_ok=True)
os.makedirs('oxford-flower/test', exist_ok=True)
os.makedirs('oxford-flower/validation', exist_ok=True)

for i in range(1,103):
    os.makedirs(f'oxford-flower/train/{label_names[i]}', exist_ok=True)
    os.makedirs(f'oxford-flower/test/{label_names[i]}', exist_ok=True)
    os.makedirs(f'oxford-flower/validation/{label_names[i]}', exist_ok=True)

for idx, imgName in enumerate(sorted(imgList)):
    
    if (idx + 1) in train_labels:
        shutil.copy(os.path.join(imgDir, imgName), os.path.join('oxford-flower', 'train', label_names[labels[idx]], imgName))
    elif (idx + 1) in test_labels:
        shutil.copy(os.path.join(imgDir, imgName), os.path.join('oxford-flower', 'test', label_names[labels[idx]], imgName))
    elif (idx + 1) in validation_labels:
        shutil.copy(os.path.join(imgDir, imgName), os.path.join('oxford-flower', 'validation', label_names[labels[idx]], imgName))

