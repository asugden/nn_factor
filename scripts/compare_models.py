import numpy as np
import sklearn.model_selection

from nn_factor import (
    factor_model_crossattn,
    factor_model_crossattn_nomlp,
    factor_model_pair_selfattn,
    factor_model_selfattn_v2,
    reader,
)


def test_model(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
) -> float:
    """Train single input model and return accuracy score"""
    if name == "cross":
        model = factor_model_crossattn.FactorModelCrossAttn()
    if name == "cross_nomlp":
        model = factor_model_crossattn_nomlp.FactorModelCrossAttnNoMLP()
    elif name == "self":
        model = factor_model_selfattn_v2.FactorModelSelfAttn()
    elif name == "selfv1":
        model = factor_model_pair_selfattn.FactorModelPairSelfAttn()
    print(model.summary())
    model.train(X_train, y_train, epochs=epochs)
    print(model.evaluate(X_test, y_test))


if __name__ == "__main__":
    epochs = 1000
    data_path = "data/formatted_symmetric_group_data.txt"

    # Read in data
    features, labels = reader.aligned_partitions(data_path)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, labels, test_size=0.1, random_state=42
    )
    test_model("self", X_train, X_test, y_train, y_test, epochs)

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
    # Self reaches 78% within 100 epochs, loss 0.4485
    #   maxes out at 88% after 519 epochs, loss 0.2704
    # Selfv1 reaches 75% within 100 epochs, loss 0.5031
    #   maxed out at least once at 93% after ~800 epochs, 88% on testing
    # Cross without MLP reaches 74% within 100 epochs, loss 0.5055
    #   stopped at ~80 epochs
