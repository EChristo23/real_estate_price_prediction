import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
import models.ml.regressor as reg
import pickle

def select_counties(df):
    df['county'] = np.where(
        df.county.isin(
            [
                'Bucuresti-Ilfov',
                'Other',
                'Cluj',
                'Iasi',
                'Brasov',
                'Constanta',
                'Sibiu',
                'Timis',
                'Braila',
                'Arad'
            ]), df.county, 'Other')
    return df


counties_transformer = FunctionTransformer(select_counties, validate=False)


def is_central(df):
    df['zona'] = np.where(
        df.zona == 'Central', 'Da', 'Nu')
    return df


zona_transformer = FunctionTransformer(is_central, validate=False)


column_transformer = make_column_transformer(
    (counties_transformer, ['county']),
    (zona_transformer, ['zona']),
    remainder='passthrough',
    verbose_feature_names_out = False
)
column_transformer.set_output(transform='pandas')


# Define feature engineering steps
def fe_area_ratio(X):
    # Your custom feature engineering logic here
    X['area_ratio'] = X['suprafata_utila']/ X['suprafata_teren']
    return X


def custom_feature_engineering(X):
    # Your custom feature engineering logic here
    X = fe_area_ratio(X)
    return X


feature_engineering_step = FunctionTransformer(func=fe_area_ratio, validate=False)

pos_processing = make_column_transformer(
    (feature_engineering_step, ['suprafata_utila', 'suprafata_teren']),
    remainder='passthrough',
    verbose_feature_names_out = False
)
pos_processing.set_output(transform='pandas')


def get_model():
    global reg
    reg.model = pickle.load(open('models/ml/rf_reg.pkl', 'rb'))