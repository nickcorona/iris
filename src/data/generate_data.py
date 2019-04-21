import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from quilt.data.uciml import iris
import pickle

PROCESSED_DATA = "data/processed"

df = iris.tables.iris()
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

lb = LabelBinarizer()
y = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)


def serialize_processed_data(name, obj, path=PROCESSED_DATA):
    with open(os.path.join(path, name + ".pickle"), "wb") as file:
        pickle.dump(obj, file)


serialize_processed_data('X_train', X_train)
serialize_processed_data('X_test', X_test)
serialize_processed_data('y_train', y_train)
serialize_processed_data('y_test', y_test)
