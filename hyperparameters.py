#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import time
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
from itertools import product
from fasta_to_csv import fas_to_csv
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.random as tfrand
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, LayerNormalization

# =============================================
def seq_tokenizer(sequencia: str) -> list:
    d = {'a': 1, 'c': 2, 'g': 3, 't': 4}
    return [d.get(i, 5) for i in sequencia]


def label_tokenizer(coluna: pd.Series) -> np.ndarray:
    d = {l:i for i,l in enumerate(coluna.unique())}
    return coluna.map(d).values


def create_model(h_param: Dict[str, list], default_params: Dict[str, int], array_shape: int) -> Sequential:
    K.clear_session()
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


# =============================================
# Helper
parser  = argparse.ArgumentParser(prog          = 'hyperparameters.py',
                                      description   = "Search hyperparameteres for model training.")

group   = parser.add_mutually_exclusive_group(required = True)

group.add_argument('-f',    '--fasta',
                    help    = '''Input fasta file with id and labels formatted as: ">seqId#Label".''')

group.add_argument('-c',    '--csv',
                    help    = '''Input CSV file containing columns "id", "label", "sequence".''')

parser.add_argument('-t',    '--title',
                    defaul  = 'TEgym',
                    type    = str,
                    help    = 'Model identifier (optional).')

parser.add_argument('-r',    '--runs',
                    default = 20,
                    type    = int,
                    help    = 'Number of runs (tests) to find the hyperparameters.')

args    = parser.parse_args()
# =============================================

# Set seed for model reproducibility
SEED = 13
# random.seed(SEED)
# np.random.seed(SEED)
tfrand.set_seed(SEED)

n_runs          = args.runs
modelo_nome     = args.title
label_column    = 'label'
timestamp       = time.strftime("%Y%m%d-%H%M%S")

if args.fasta:
    infasta = Path(args.fasta)
    fas_to_csv(infasta)
    arquivo = Path(f"{infasta.stem}.csv")
elif args.csv:
    arquivo = Path(args.csv)

basename    = arquivo.stem
saida       = f"hyperparams_{modelo_nome}_on_{basename}_in_{timestamp}.csv"
df          = pd.read_csv(arquivo, usecols=['sequence',label_column])


# compute weights
rotulos         = label_tokenizer(df[label_column])
rotulos_unicos  = np.unique(rotulos)
pesos = compute_class_weight(
    class_weight    = 'balanced',
    classes         = rotulos_unicos,
    y               = rotulos
)
pesos_dict = dict(zip(rotulos_unicos, pesos))


parameters      = {'conv1':          [32,   64,     128],
                   'convN':          [24,   32,     64],
                   'kernel_size':    [6,    12],
                   'pool_size':      [3,    6],
                   'dense_neuron':   [32,   64,     128],
                   'dropout':        [0.0,  0.2,    0.4],
                   'batch_size':     [25,   50,     75],
                   'epochs':         [15,   20]}

default_params  = {'emb_dim':           6,
                   'seq_len':           25_000,
                   'val_split':         0.1,
                   'vocab_size':        6,
                   'vocab_size_lab':    len(df[label_column].unique()),
                   'weights':           pesos_dict
                   }

combinations = list(product(*parameters.values()))
h_params = random.sample(combinations, n_runs)

padded_seqs = pad_sequences(df.sequence.map(seq_tokenizer).values,
                            padding     = 'post',
                            truncating  = 'post',
                            maxlen      = default_params['seq_len'])
labels      = to_categorical(label_tokenizer(df[label_column]))
in_shape    = padded_seqs.shape[1]
del df
gc.collect()

tamanho_teste = default_params["val_split"]
x_train, x_test, y_train, y_test = train_test_split(padded_seqs,
                                                    labels,
                                                    test_size       = tamanho_teste,
                                                    random_state    = SEED,
                                                    stratify        = labels,
                                                    shuffle         = True)
del padded_seqs, labels

with open(saida, "w+") as sd:
    keys = list(parameters.keys())
    header = ",".join([*parameters.keys(),
                       *default_params.keys(),
                       "loss",
                       "accuracy",
                       "val_loss",
                       "val_accuracy"])
    sd.write(f'{header}\n')
    for i, h_param in enumerate(h_params):
        i += 1
        h_param = dict(zip(keys, h_param))
        print("###############")
        print(f"### Round {i:2} ###")
        print("###############")
        print(*[str(k) + ": " + str(v) for k, v in h_param.items()],
              sep=", ",
              end=".")
        print("\n")

        model = create_model(h_param,
                             default_params,
                             in_shape)

        # Fit the model
        history = model.fit(x_train,
                            y_train,
                            epochs              = h_param['epochs'],
                            batch_size          = h_param['batch_size'],
                            validation_data     = (x_test, y_test),
                            verbose             = 1,
                            shuffle             = True,
                            class_weight        = pesos_dict
                            )

        metrics = {k: v[-1] for k, v in history.history.items()}
        K.clear_session()
        gc.collect()
        h_param.update(default_params)
        h_param.update(metrics)
        sd.write(f"{','.join([str(i) for i in h_param.values()])}\n")
        sd.flush()
