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
    def create_model(self, params_dict: Dict[str, list], array_shape: int) -> Sequential:
        model = Sequential()

        model.add(Embedding(input_dim       = params_dict["vocab_size"],
                            output_dim      = params_dict['emb_dim'],
                            input_length    = array_shape,
                            mask_zero       = False))

        model.add(Conv1D(filters            = params_dict['conv1'],
                         kernel_size        = params_dict['kernel_size'],
                         input_shape        = (array_shape, 1),
                         activation         = 'relu'))
        model.add(MaxPooling1D(pool_size    = params_dict['pool_size']))

        model.add(Conv1D(filters            = params_dict['convN'],
                         kernel_size        = params_dict['kernel_size'],
                         activation         = 'relu'))
        model.add(MaxPooling1D(pool_size    = params_dict['pool_size']))

        model.add(Conv1D(filters            = params_dict['convN'],
                         kernel_size        = params_dict['kernel_size'],
                         activation         = 'relu'))
        model.add(MaxPooling1D(pool_size    = params_dict['pool_size']))

        model.add(LayerNormalization())
        model.add(Flatten())

        model.add(Dense(params_dict['dense_neuron'],
                        activation = 'relu'))

        model.add(Dropout(params_dict["dropout"]))

        model.add(Dense(params_dict["vocab_size_lab"],
                        activation = 'softmax'))

        model.compile(loss      = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics   = ['accuracy'])

        return model

    def model_fit(self, model, params_dict, X_train, y_train, X_test, y_test, c_weights):
        history = model.fit(X_train,
                            y_train,
                            epochs          = params_dict['epochs'],
                            batch_size      = params_dict['batch_size'],
                            validation_data = (X_test, y_test),
                            verbose         = 1,
                            shuffle         = True,
                            class_weight    = c_weights)
        return history
