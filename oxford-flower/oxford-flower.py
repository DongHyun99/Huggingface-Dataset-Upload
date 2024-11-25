import os
from datasets import Dataset, ClassLabel, DatasetDict, Features, Image
from huggingface_hub import notebook_login

notebook_login()

img_dir = 'jpg'
label = 'label.txt'
label_name = 'label_name.txt'
trainval_label = 'train.txt'
test_label = 'test.txt'
validation_label = 'validation.txt'

labels = open(label, 'r').read().split(', ')
label_names = open(label_name, 'r').read().split(', ')
trainval_labels = open(trainval_label, 'r').read().split(', ')
test_labels = open(test_label, 'r').read().split(', ')
validation_labels = open(validation_label, 'r').read().split(', ')

split = [0 if str(idx+1) in trainval_labels else 1 if str(idx+1) in validation_labels else 2 for idx in range(len(labels))]

print(str(2) in test_labels)

data = {
    "image": [f'jpg/{path}' for path in sorted(os.listdir('jpg'))],
    "label": [int(l)-1 for l in labels],
    "split": split
}

features = Features({
    "image": Image(decode=True),
    "label": ClassLabel(names=label_names),
    "split": ClassLabel(names=["train", "validation", "test"])
})

dataset = Dataset.from_dict(data, features=features)

dataset_dict = DatasetDict({
    "train": dataset.filter(lambda x: x["split"] == 0), 
    "validation": dataset.filter(lambda x: x["split"] == 1),
    "test": dataset.filter(lambda x: x["split"] == 2)
    }).remove_columns("split")

dataset_dict.push_to_hub("Donghyun99/Oxford-Flower-102")