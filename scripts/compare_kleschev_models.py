from nn_factor import (
    kleschev_cnn_model,
    kleschev_crossattn_model,
    kleschev_selfattn_model,
)

if __name__ == "__main__":
    # model = kleschev_cnn_model.KleschevCNNModel()
    # model = kleschev_selfattn_model.KleschevSelfAttnModel(
    #     k_segment_as_multiplicative=True
    # )
    model = kleschev_crossattn_model.KleschevCrossAttnModel(
        k_segment_as_multiplicative=True
    )
    model.summary()
