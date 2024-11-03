import pandas as pd

def prepare_features_and_labels(transformed_data, transform_config):
    '''
    Below code creates a feature matrix and a target vector,
    from selected columns in the dataset, which are ready to be used for model training.
    '''

    features = transformed_data[transform_config['feature_column_names']].to_numpy()
    labels = transformed_data[transform_config['label_column_name']].to_numpy()

    print("Features:")
    print(pd.DataFrame(features, columns=[transform_config['feature_column_names']]), "\n")

    print("Labels: {}".format(transform_config['label_column_name']))
    print(pd.Series(labels, name=transform_config['label_column_name']))

    return features, labels