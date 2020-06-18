import preprocessing_functions as pf
import config

import warnings
warnings.simplefilter(action='ignore')

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
data = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(data, config.TARGET)


# get first letter from cabin variable
X_train["cabin"] = pf.extract_cabin_letter(X_train, "cabin")
print(X_train["cabin"].unique())


# impute categorical variables
for var in config.CATEGORICAL_VARS:
    X_train[var] = pf.impute_na(X_train, var, replacement='Missing')



# impute numerical variable with median
for var in config.NUMERICAL_TO_IMPUTE:
    X_train[var+"_na"] = pf.add_missing_indicator(X_train, var)
    median_train_var = config.IMPUTATION_DICT[var]
    X_train[var] = pf.impute_na(X_train, var, replacement=median_train_var)


# Group rare labels
for var in config.CATEGORICAL_VARS:
     # Frequent labels found in Train set
    freq_labels = config.FREQUENT_LABELS[var]
    # Remove rare labels from both train and test set
    X_train[var] = pf.remove_rare_labels(X_train, var, freq_labels)


# encode categorical variables
for var in config.CATEGORICAL_VARS:
    X_train = pf.encode_categorical(X_train, var)


# check all dummies were added
X_train = pf.check_dummy_variables(X_train, config.DUMMY_VARIABLES)
    


# train scaler and save
scaler = pf.train_scaler(X_train, config.OUTPUT_SCALER_PATH)


# scale train set
X_train = pf.scale_features(X_train, config.OUTPUT_SCALER_PATH)


# train model and save
pf.train_model(X_train,
               y_train,
               config.OUTPUT_MODEL_PATH)


print('Finished training')