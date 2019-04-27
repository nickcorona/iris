from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_pickle('data/raw/iris.pickle')


X = pd.read_pickle('data/processed/X_train.pickle')  #