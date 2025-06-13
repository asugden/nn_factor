from nn_factor import (
    kleshchev_cnn_model,
    kleshchev_crossattn_model,
    kleshchev_selfattn_model,
)

if __name__ == "__main__":
    # model = kleschev_cnn_model.KleschevCNNModel()
    # model = kleschev_selfattn_model.KleschevSelfAttnModel(
    #     k_segment_as_multiplicative=True
    # )
    model = kleshchev_crossattn_model.KleshchevCrossAttnModel(
        k_segment_as_multiplicative=True
    )
    model.summary()
