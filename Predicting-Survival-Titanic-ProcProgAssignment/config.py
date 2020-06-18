# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===============

# imputation parameters
IMPUTATION_DICT = {
    'fare': 14.4542,
    'age': 28.0
}


# encoding parameters
FREQUENT_LABELS = {
    'sex': ['female', 'male'],
    'cabin': ['C', 'Missing'],
    'embarked': ['C', 'Q', 'S'],
    'title': ['Miss', 'Mr', 'Mrs']
}


DUMMY_VARIABLES = [ 'sex_male',
                    'cabin_Missing', 'cabin_Rare',
                    'embarked_Q', 'embarked_Rare', 'embarked_S',
                    'title_Mr', 'title_Mrs', 'title_Rare'
]


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_TO_IMPUTE = ['age', 'fare']