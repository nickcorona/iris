# -*- coding: utf-8 -*-
import os
import pickle
from quilt.data.uciml import iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


df = iris.tables.iris()
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

lb = LabelBinarizer()
y = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)


def serialize_object(name, obj, path):
    with open(os.path.join(path, name + ".pickle"), "wb") as file:
        pickle.dump(obj, file)


data_path = "data/processed"

serialize_object("X_train", X_train, data_path)
serialize_object("X_test", X_test, data_path)
serialize_object("y_train", y_train, data_path)
serialize_object("y_test", y_test, data_path)
serialize_object("feature_names", list(df.columns[0:4]), data_path)
print("Saved pickles.")
