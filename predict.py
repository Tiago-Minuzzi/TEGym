import os
import sys
import time
import toml
import pandas as pd
from numpy import argmax
from pathlib import Path
from Bio.SeqIO.FastaIO import SimpleFastaParser

# Hide warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from preprocessing import Preprocessor
from tensorflow.keras.models import load_model
sys.stderr = stderr

input_fasta     = Path(sys.argv[1])
timestamp       = time.strftime('%y%m%d%H%M%S')
output_table    = f"{input_fasta.stem}_{timestamp}_prediction.csv"
model           = load_model(sys.argv[2])
model_info      = Path(sys.argv[3])

pp = Preprocessor()

with open(input_fasta) as fa, open(model_info) as tmf:
    fids = []
    fsqs = []
    print("### Loading sequences ###")
    for fid, fsq in SimpleFastaParser(fa):
        fsq = fsq.lower()
        fids.append(fid)
        fsqs.append(fsq)

    tmf         = toml.load(tmf)
    tmf_dict    = tmf['model_info']
    labels      = tmf_dict['Labels']
    labels_dict = {k: v for k, v in enumerate(labels)}
    seq_len     = tmf_dict['Maximum_length']

    print("### Tokenizing and padding sequences ###\n")
    fsqs = pp.zero_padder(list(map(pp.seq_tokenizer, fsqs)),
                          seq_len)

    print("### Starting prediction ###")
    predictions = model.predict(fsqs)
    pred_labels = argmax(predictions, axis=1)
    df          = pd.DataFrame(predictions, columns=labels).round(4)
    df.insert(0, 'id', pd.Series(fids))
    df.insert(1, 'prediction', pd.Series(pred_labels))
    df['prediction'] = df['prediction'].map(labels_dict)
    df.to_csv(output_table, index=False)
    print(">>> Done!")
