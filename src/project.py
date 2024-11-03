import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

from dataset.titanic import get_titanic_data, transform_titanic_data
from models.perceptron.network import input_layer

## Fetch dataset
titanic_data_config = get_titanic_data()

file_path = os.path.join(titanic_data_config['download_path'], titanic_data_config['filename'])
if not os.path.exists(file_path):
        print(f"File does not exists at {file_path}. Downloading...")
        file_path = download_dataset(titanic_data_config['url'], titanic_data_config['download_path'], titanic_data_config['filename'])


## Read data
np.random.seed(10)

data = pd.read_csv(file_path)
print("Data (first 4 rows):")
print(data.head(4), "\n")

## Prepare input layer
transformed_data, transform_config = transform_titanic_data(data)
features, labels = input_layer.prepare_features_and_labels(transformed_data, transform_config)

## split into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30)

print('Training records:',Y_train.size)
print('Test records:',Y_test.size)