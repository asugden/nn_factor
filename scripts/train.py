import numpy as np
import sklearn.model_selection

from nn_factor import factor_model


def read_data(path: str) -> tuple[np.array, np.array]:
    """Read in data from Corinne"""
    features, labels = [], []
    with open(path, "r") as fp:
        for line in fp.readlines():
            sections = line.split("][")
            if len(sections) == 2:
                features.append([int(v) for v in sections[0][1:].split(",")])
                labels.append(int(sections[1].split("]")[0]))
    return np.array(features), np.array(labels)


if __name__ == "__main__":
    features, labels = read_data("data/formatted_symmetric_group_data.txt")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, labels, test_size=0.1, random_state=42
    )
    model = factor_model.FactorModel()
    model.train(X_train, y_train, epochs=200)
    print(model.evaluate(X_test, y_test))
    print(model.save("data/trained_model_200.tf"))
    print(model.summary())
