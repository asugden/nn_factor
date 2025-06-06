import numpy as np
import sklearn.model_selection

from nn_factor import factor_model_v2, reader

if __name__ == "__main__":
    features, labels = reader.aligned_partitions(
        "data/formatted_symmetric_group_data.txt"
    )
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, labels, test_size=0.1, random_state=42
    )
    model = factor_model_v2.FactorModel()
    model.train(X_train, y_train, epochs=1000)
    print(model.evaluate(X_test, y_test))
    # print(model.save("data/trained_model_v2_100.tf"))

    pred = model.predict(X_test)
    correct = (pred >= 0.5).astype(int) == y_test.astype(int)
    print(correct)
