import os
from datasets import Dataset, ClassLabel, DatasetDict, Features, Image
from huggingface_hub import notebook_login

notebook_login()

img_dir = 'images'
trainval_label = 'annotations/trainval.txt'
test_label = 'annotations/test.txt'

trainval_labels = open(trainval_label, 'r').read()
test_labels = open(test_label, 'r').read()

trainval_labels = [label.split(' ') for label in trainval_labels.split('\n')][:-1]
test_labels = [label.split(' ') for label in test_labels.split('\n')][:-1]

label = {int(label[1])-1: ' '.join(label[0].split('_')[:-1]) for label in trainval_labels}
species = {int(label[2])-1: ' '.join(label[0].split('_')[:-1]) for label in trainval_labels}
breed = {int(label[3])-1: ' '.join(label[0].split('_')[:-1]) for label in trainval_labels}

data = {
    "image": [f'images/{path[0]}.jpg' for path in trainval_labels]+[f'images/{path[0]}.jpg' for path in test_labels],
    "label": [int(path[1])-1 for path in trainval_labels]+[int(path[1])-1 for path in test_labels],
    "species": [int(path[2])-1 for path in trainval_labels]+[int(path[2])-1 for path in test_labels],
    "split": [0]*len(trainval_labels)+[1]*len(test_labels)
}

features = Features({
    "image": Image(decode=True),
    "label": ClassLabel(names=[label[idx] for idx in range(0,len(label))]),
    "species": ClassLabel(names=["Cat", "Dog"]),
    "split": ClassLabel(names=["train", "test"])
})

dataset = Dataset.from_dict(data, features=features)
dataset_dict = DatasetDict({
    "train": dataset.filter(lambda x: x["split"] == 0), 
    "test": dataset.filter(lambda x: x["split"] == 1)
    }).remove_columns("split")

dataset_dict.push_to_hub("Donghyun99/Oxford-IIIT-Pet")