import os
import sys
import time
import argparse
import pandas as pd
from pathlib import Path
# Hide warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from preprocessing import Sampler
sys.stderr = stderr
from fasta_to_csv import fas_to_csv

# ==============================================

parser = argparse.ArgumentParser(prog='create_negative_class.py',
                                 description='Create a negative class and join to input dataset to train a model.')

parser.add_argument('-f', '--fasta',
                    required = True,
                    help="Input FASTA file with headers/ids in the RepeatMasker format.")

parser.add_argument('-c', '--create',
                    default='shuffled',
                    help="Create random or shuffled sequences to be used as negative class. Values: random or shuffled.")


args = parser.parse_args()

# ==============================================

sampler = Sampler()

# ==============================================

timestamp   = time.strftime('%y%m%d-%H%M%S')
input_fasta = Path(args.fasta)
input_csv   = Path(f"{input_fasta.stem}.csv")
output_tab  = f"TDS_{input_fasta.stem}_{timestamp}.csv"

# ==============================================

create_sequences = args.create

# ==============================================
print("### Starting program ###")
print("### Converting FASTA to CSV ###")
fas_to_csv(input_fasta)

if input_csv.exists():
    df          = pd.read_csv(input_csv)
    n_seqs      = df.shape[0]
    min_size    = df['sequence'].str.len().min()
    max_size    = df['sequence'].str.len().max()

    copia       = df.copy()

    print("### Creating sequences ###")
    if create_sequences == 'shuffled':
        copia['sequence']   = copia['sequence'].map(sampler.seq_shuffler)
        copia['id']         = 'shuffled_' + copia['id']
        copia['label']      = 'Other'

    elif create_sequences == 'random':
        random_seqs = \
        sampler.create_random_sequences(n_seqs  = n_seqs,
                                        lmin    = min_size,
                                        lmax    = max_size)
        copia['sequence']   = [*random_seqs]
        copia['id']         = 'random_' + copia.index.astype(str)
        copia['label']      = 'Other'

    elif create_sequences not in ['random', 'shuffled']:
        print(">>> Warning:")
        print('>>> Invalid value for "--create". Using default (shuffled).\n')
        metric = 'shuffled'

    print("### Writing to file ###")
    df = pd.concat([df, copia], ignore_index=True)
    df.to_csv(output_tab, index=False)

    print(">>> Done!")
    print(f">>> Saved as {output_tab}")

