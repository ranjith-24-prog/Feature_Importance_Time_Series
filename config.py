import prepare_data

features_data, targets_data = prepare_data.prepare_data()


LSTM = {
        "PI_Param": {
            "input_size": 52,
            "hidden_size": 64,
            "output_size": 4,
            "num_layers": 2
        },
        "IG_Param": {
            "input_size": 52,
            "hidden_size": 64,
            "output_size": 4,
            "num_layers": 2
        },
        "LIME_Param": {
            "input_size": features_data.shape[1],
            "hidden_size": 64,
            "num_layers": 2
        }
    }