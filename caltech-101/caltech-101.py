from datasets import load_dataset
from huggingface_hub import notebook_login

notebook_login()

dataset = load_dataset("caltech-101")
dataset = dataset['train'].train_test_split(test_size=0.6627, stratify_by_column="label")

dataset.push_to_hub("Donghyun99/caltech-101")