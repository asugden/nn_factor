from nn_factor import width_cnn_model, width_selfattn_model
from nn_factor.readers import width

if __name__ == "__main__":
    import sklearn.model_selection

    epochs = 20
    data_path = "data/width_data.csv"

    # Read in data
    features, labels = width.read_width(data_path)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, labels, test_size=0.1, random_state=42
    )

    # model = width_selfattn_model.WidthSelfAttnModel()
    model = width_cnn_model.WidthCNNModel()
    print(model.summary())
    model.train(
        X_train,
        y_train,
        epochs=epochs,
        loss="mean_absolute_error",
        metrics=[],
        batch_size=2048,
    )
    print(model.evaluate(X_test, y_test))
