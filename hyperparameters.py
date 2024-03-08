#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import os
import sys
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
# Hide warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from neural_net import Trainer
from preprocessing import Preprocessor
import tensorflow.random as tfrand
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
sys.stderr = stderr
pd.options.mode.chained_assignment = None  # default='warn'

# =============================================

trainer         = Trainer()
preprocessor    = Preprocessor()

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
                    default = 'TEgym',
                    type    = str,
                    help    = 'Model identifier (optional).')

parser.add_argument('-r',    '--runs',
                    default = 20,
                    type    = int,
                    help    = 'number of runs (tests) to find the hyperparameters.')

parser.add_argument('-s',    '--split',
                    default = 0.1,
                    type    = float,
                    help    = 'Portion of the dataset to use as validation set. The major portion is used for model training. Default=0.1.')

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

print('########################')
print('###     Welcome      ###')
print('### Starting program ###')
print('########################\n')

basename    = arquivo.stem
saida       = f"hyperparams_{modelo_nome}_on_{basename}_in_{timestamp}.csv"
df          = pd.read_csv(arquivo, usecols=['sequence',label_column])


# compute weights
pesos_dict = preprocessor.get_weight(df[label_column])

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

padded_seqs = preprocessor.zero_padder(df.sequence.map(preprocessor.seq_tokenizer).values,
                                       default_params['seq_len'])

labels      = preprocessor.transform_label(df[label_column])
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
        print("################")
        print(f"### Round {i:2} ###")
        print("################")
        print(*[str(k) + ": " + str(v) for k, v in h_param.items()],
              sep=", ",
              end=".")
        print("\n")

        model = trainer.create_model(h_param,
                                     default_params,
                                     in_shape)

        # Fit the model
        history = trainer.model_fit(model       = model,
                                    X_train     = x_train,
                                    y_train     = y_train,
                                    n_epochs    = h_param['epochs'],
                                    n_batch     = h_param['batch_size'],
                                    X_test      = x_test,
                                    y_test      = y_test,
                                    c_weights   = pesos_dict)

        metrics = {k: v[-1] for k, v in history.history.items()}
        gc.collect()
        h_param.update(default_params)
        h_param.update(metrics)
        sd.write(f"{','.join([str(i) for i in h_param.values()])}\n")
        sd.flush()
