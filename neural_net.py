import os
import sys
from typing import Dict
# Hide warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, Dropout, LayerNormalization
sys.stderr = stderr


class Trainer:
    def create_model(self, h_param: Dict[str, list], default_params: Dict[str, int], array_shape: int) -> Sequential:
        model = Sequential()

        model.add(Embedding(input_dim       = default_params["vocab_size"],
                            output_dim      = default_params['emb_dim'],
                            input_length    = array_shape,
                            mask_zero       = False))

        model.add(Conv1D(filters            = h_param['conv1'],
                         kernel_size        = h_param['kernel_size'],
                         input_shape        = (array_shape, 1),
                         activation         = 'relu'))
        model.add(MaxPooling1D(pool_size    = h_param['pool_size']))

        model.add(Conv1D(filters            = h_param['convN'],
                         kernel_size        = h_param['kernel_size'],
                         activation         = 'relu'))
        model.add(MaxPooling1D(pool_size    = h_param['pool_size']))

        model.add(Conv1D(filters            = h_param['convN'],
                         kernel_size        = h_param['kernel_size'],
                         activation         = 'relu'))
        model.add(MaxPooling1D(pool_size    = h_param['pool_size']))

        model.add(LayerNormalization())
        model.add(Flatten())

        model.add(Dense(h_param['dense_neuron'],
                        activation = 'relu'))

        model.add(Dropout(h_param["dropout"]))

        model.add(Dense(default_params["vocab_size_lab"],
                        activation = 'softmax'))

        model.compile(loss      = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics   = ['accuracy'])

        return model

    def model_fit(self, model, X_train, y_train, X_test, y_test, n_epochs, n_batch, c_weights):
        history = model.fit(X_train,
                            y_train,
                            epochs          = n_epochs,
                            batch_size      = n_batch,
                            validation_data = (X_test, y_test),
                            verbose         = 1,
                            shuffle         = True,
                            class_weight    = c_weights)
        return history
