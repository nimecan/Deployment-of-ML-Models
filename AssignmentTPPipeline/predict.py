import pandas as pd

import joblib
import config


def make_prediction(input_data):
    
    # load pipeline and make predictions
    _pipe_price = joblib.load(filename=config.PIPELINE_NAME)

    results = _pipe_price.predict(input_data)

    # rturn predictions
    return results
   
if __name__ == '__main__':
    
    # test pipeline
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = pd.read_csv(config.TRAINING_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.CATEGORICAL_VARS+config.NUMERICAL_VARS],
        data[config.TARGET],
        test_size=0.2,
        random_state=0)  # we are setting the seed here
    
    pred = make_prediction(X_test)
    
    # determine the accuracy
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()

