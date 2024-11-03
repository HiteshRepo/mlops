import os
import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset.download_dataset import download_dataset

## Fetch dataset
url = "https://www.kaggle.com/api/v1/datasets/download/aibuzz/titanic-train-dataset" # not a valid url, directly downloaded the file
data_dir = "data/kaggle"
data_file = "train.csv"

file_path = os.path.join(data_dir, data_file)
if not os.path.exists(file_path):
        print(f"File does not exists at {file_path}. Downloading...")
        file_path = download_dataset(url, data_dir, data_file)


## Read data
np.random.seed(10)

data = pd.read_csv(file_path)
print(data.head(4))