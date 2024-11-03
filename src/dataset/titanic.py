def get_titanic_data():
    # not a valid url, directly downloaded the file
    url = "https://www.kaggle.com/api/v1/datasets/download/aibuzz/titanic-train-dataset" 
    data_dir = "data/kaggle"
    data_file = "train.csv"

    data_config = {}
    data_config['url'] = url
    data_config['download_path'] = data_dir
    data_config['filename'] = data_file

    return data_config

def transform_titanic_data(data):
    dict_live = { 
        0 : 'Perished',
        1 : 'Survived'
    }

    dict_sex = {
        'male' : 0,
        'female' : 1
    }

    data['Bsex'] = data['Sex'].apply(lambda x : dict_sex[x])

    transform_config = {}
    transform_config['feature_column_names'] = ['Pclass', 'Bsex']
    transform_config['label_column_name'] = 'Survived'

    return data, transform_config