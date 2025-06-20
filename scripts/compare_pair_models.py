from typing import Literal

import numpy as np
import sklearn.model_selection

from nn_factor import partition_pair_model
from nn_factor.readers import pair


def test_model(
    attention_type: Literal["cross", "self"],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    skip_mlp: bool = False,
    activation_fn: Literal["relu", "gelu"] = "gelu",
) -> float:
    """Train single input model and return accuracy score"""
    model = partition_pair_model.PartitionPairModel(
        attention_type=attention_type, skip_mlp=skip_mlp, activation_fn=activation_fn
    )
    print(model.summary())
    model.train(X_train, y_train, epochs=epochs)
    print(model.evaluate(X_test, y_test))


if __name__ == "__main__":
    epochs = 20
    data_path = "data/formatted_symmetric_group_data.txt"

    # Read in data
    features, labels = pair.aligned_partitions(data_path)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, labels, test_size=0.1, random_state=42
    )
    test_model("cross", X_train, X_test, y_train, y_test, epochs)

    # Single input maxes out at 73% performance
    # Note that it is trained on just partitions, not aligned_partitions
    # test_single_input(X_train, X_test, y_train, y_test, epochs)

    # Data is correctly read in for later models
    # Pair self-attention model maxes out at 88% performance and
    # overtrains when done to 1000 epochs. In that case, performance
    # on the training data reached 93% accuracy.
    # test_model('selfv1', X_train, X_test, y_train, y_test, epochs)

    # Cross reaches 78% within 100 epochs, loss 0.4590
    #   maxes out at 83% after 354 epochs, loss 0.3554
    #   switching to gelu maxes out at 93% after 815 epochs, loss 0.1683,
    #   test set gets 91.4% accuracy
    # Self reaches 78% within 100 epochs, loss 0.4485
    #   maxes out at 88% after 519 epochs, loss 0.2704
    # Selfv1 reaches 75% within 100 epochs, loss 0.5031
    #   maxed out at least once at 93% after ~800 epochs, 88% on testing
    # Cross without MLP reaches 74% within 100 epochs, loss 0.5055
    #   stopped at ~80 epochs
