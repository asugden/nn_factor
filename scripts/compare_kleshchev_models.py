from typing import Literal

import numpy as np
import sklearn.model_selection

from nn_factor import (
    kleshchev_cnn_model,
    kleshchev_crossattn_model,
    kleshchev_selfattn_model,
)
from nn_factor.readers import multicharge

if __name__ == "__main__":
    epochs = 20
    data_path = "data/final_fixed_20_small.txt"

    # Read in data
    mcs, mps, labels = multicharge.read_multicharge(data_path)
    features = multicharge.convert_cross_attention(mcs, mps)
    # features = multicharge.convert_self_attention(mcs, mps)

    # split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, labels, test_size=0.1, random_state=42
    )

    model = kleshchev_cnn_model.KleshchevCNNModel()
    # model = kleshchev_selfattn_model.KleshchevSelfAttnModel(
    #     k_segment_as_multiplicative=True
    # )
    # model = kleshchev_crossattn_model.KleshchevCrossAttnModel(
    #     k_segment_as_multiplicative=True, max_M=10, max_N=20
    # )
    print(model.summary())
    model.train(X_train, y_train, epochs=epochs)
    print(model.evaluate(X_test, y_test))
    model.summary()
