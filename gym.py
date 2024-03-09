import argparse
import pandas as pd
from pathlib import Path
from neural_net import Trainer
from preprocessing import Preprocessor

# =============================================

tr = Trainer()
pp = Preprocessor()

# =============================================

# Helper
parser  = argparse.ArgumentParser(prog          = 'gym.py',
                                  description   = "Train your own classifier model.")

group   = parser.add_mutually_exclusive_group(required = True)

group.add_argument('-f',    '--fasta',
                    help    = '''Input fasta file with id and labels formatted as: ">seqId#Label".''')

group.add_argument('-c',    '--csv',
                    help    = '''Input CSV file containing columns "id",
"label", "sequence".''')

parser.add_argument('-p',    '--hyper',
                    help    = 'CSV file containing the hyperparametere metrics')

parser.add_argument('-m',    '--metric',
                    default = 'val_loss',
                    help    = 'choose hyperparameters based on metric. Values are "val_loss" (default) or "val_accuracy".')

parser.add_argument('-t',    '--title',
                    default = 'TEgym',
                    type    = str,
                    help    = 'Model identifier (optional).')

parser.add_argument('-s',    '--split',
                    default = 0.1,
                    type    = float,
                    help    = 'Portion of the dataset to use as validation set. The major portion is used for model training. Default=0.1')

args    = parser.parse_args()
# =============================================
# Files
hyper_csv = Path(args.hyper)

# =============================================
# Variables
metric      = args.metric
sort_order  = True
# =============================================
if hyper_csv:
    hyper_df = pd.read_csv(hyper_csv)
    if metric == 'val_accuracy':
        sort_order  = False
    elif metric not in ['val_loss', 'val_accuracy']:
        print(">>> Warning:")
        print('>>> Invalid value for "--metric". Using default (val_loss).\n')
        metric = 'val_loss'
    hyper_df.sort_values(metric, inplace=True, ascending=sort_order)
    hp_dict = hyper_df.head(1).to_dict('records')[0]
