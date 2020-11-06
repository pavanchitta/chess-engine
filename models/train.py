####
# Train and save a model
####

import numpy as np
import data_processing as dp
import neural_net


if __name__ == "__main__":


    X, y, weights = dp.load_data(ngames=1e6)
    # X = np.expand_dims(X, axis=-1)
    nn_mdl = neural_net.NeuralNet(input_dim=(8, 8, 6))

    nn_mdl.model.fit(X, y, batch_size=64, epochs=10, verbose=1,
                     validation_split=0.2, sample_weight=weights)

