####
# Train and save a model
####

import numpy as np
import data_processing as dp
import neural_net


if __name__ == "__main__":

    ngames = int(5 * 1e5)
    NAME = 'all_ratings'
    X, y, weights = dp.load_data(ngames=ngames, use_cache=True, name=NAME)
    nn_mdl = neural_net.NeuralNet(input_dim=(8, 8, 7))

    batch_size = 512
    epochs = 5
    val_split = 0.2

    nn_mdl.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_split=val_split, sample_weight=weights)

    nn_mdl.model.save(f'trained_models/{NAME}_{ngames}_bsize{batch_size}_epochs{epochs}')
