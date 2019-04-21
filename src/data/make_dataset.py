# -*- coding: utf-8 -*-
import logging
import os
import pickle
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from quilt.data.uciml import iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    os

    df = iris.tables.iris()
    X = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values

    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    def serialize_processed_data(name, obj, path=output_filepath):
        with open(os.path.join(path, name + ".pickle"), "wb") as file:
            pickle.dump(obj, file)

    serialize_processed_data("X_train", X_train)
    serialize_processed_data("X_test", X_test)
    serialize_processed_data("y_train", y_train)
    serialize_processed_data("y_test", y_test)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main("/data/processed/")
