import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)


def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),
                                                        df[target],
                                                        test_size=0.2,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test



def extract_cabin_letter(df, var):
    # captures the first letter
    return df[var].str[0] 



def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    return np.where(df[var].isnull,1,0)
    

    
def impute_na(df, var, replacement='Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df[var].fillna(replacement)



def find_frequent_labels(df, var, target, rare_perc, frequent_label_list):
    # function finds the labels that are shared by more than
    # a certain % of rows in the dataset
    df = df.copy()
    tmp = df.groupby(var)[target].count() / len(df)
    
    return tmp[tmp > rare_perc].index



def remove_rare_labels(df, var, frequent_labels):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')



def encode_categorical_ohe(df, var_list):
    # adds ohe variables and removes original categorical variable
    df = df.copy()
    
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(categories='auto',
                        drop='first', # to return k-1, use drop=false to return k dummies
                        sparse=False,
                        handle_unknown='error') # helps deal with rare labels

    encoder.fit(pd.DataFrame(df.loc[:,var_list]))
    # Transform dataframe with enconder
    tmp = pd.DataFrame(encoder.transform(pd.DataFrame(df.loc[:,var_list])), index=df.index)
    # New var names
    tmp.columns = encoder.get_feature_names()
    # Remove original variables or columns
    df.drop(var_list, axis=1, inplace=True)
    # Add new encoded variables
    df = pd.concat([df, tmp], axis=1)

    return df

def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    df = df.copy()
    df = pd.concat([df,
                    pd.get_dummies(df[var], prefix=var, drop_first=True)
                ], axis=1)

    df.drop(var, axis=1, inplace=True)
    return df
    



def check_dummy_variables(df, dummy_list):
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    for var in dummy_list:
        if not var in df.columns:
            df[var] = 0
       
    return df
    

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler
  
    

def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path) # with joblib probably
    return scaler.transform(df)



def train_model(df, target, output_path):
    # initialise the model
    classifier = LogisticRegression(C=0.0005, random_state=0)
    
    # train the model
    classifier.fit(df, target)
    
    # save the model
    joblib.dump(classifier, output_path)
    
    return None



def predict(df, model):
    # load model and get predictions
    model = joblib.load(model)
    return model.predict(df)

def predict_proba(df, model):
    model = joblib.load(model)
    return model.predict_proba(df)

