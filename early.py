import os
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer, ColumnTransformer,make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


TARGET = 'Survived'
TEST_SIZE = 0.33

#%% Load data
raw = pd.read_csv('https://gist.githubusercontent.com/Takfes/cc7ae79cf334a7d33089c6a126834848/raw/f38e0a8646ff0607c240bcfa96819e943b26ed63/titanic.csv').set_index('PassengerId').dropna()

def prepare_data(df):
    columns_to_keep = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked', 'Survived']
    dataframe = df[columns_to_keep].copy()
    embarked_map = {'C' : 0, 'Q' : 1, 'S' : 2}
    dataframe['Embarked'] = dataframe['Embarked'].map(embarked_map)
    dataframe['Parch_Sibsp'] = dataframe['Parch'] + dataframe['SibSp']
    return dataframe

df = prepare_data(raw)
X = df.drop(TARGET,axis=1)
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1990, stratify = y)

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Parch_Sibsp']
categorical_features = ['Embarked', 'Sex', 'Pclass']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

pipe = Pipeline([('preprocessor', preprocessor),('RF',RandomForestClassifier())])
cross_val_score(pipe, X, y, scoring = 'roc_auc', cv = 10)


def fn_embarked(df):
    dataframe = df.copy()
    embarked_map = {'C' : 0, 'Q' : 1, 'S' : 2}
    dataframe['Embarked'] = dataframe['Embarked'].map(embarked_map)
    return dataframe

def fn_parch_sibsp(df):
    dataframe = df.copy()
    dataframe['Parch_Sibsp'] = dataframe['Parch'] + dataframe['SibSp']
    return dataframe

trns_emabrked = FunctionTransformer(fn_embarked, validate=False)
trns_parch_sibsp = FunctionTransformer(fn_parch_sibsp, validate=False)

pp1 = Pipeline(memory=None,
    steps=[
        ('embarked', trns_emabrked),
        ('parch_sibsp', trns_parch_sibsp)
    ], verbose=False)

pp1.fit_transform(raw)