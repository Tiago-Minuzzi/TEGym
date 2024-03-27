import time
import argparse
import subprocess
import pandas as pd
from shutil import move
from pathlib import Path
from neural_net import Trainer
from fasta_to_csv import fas_to_csv
from preprocessing import Preprocessor
from sklearn.model_selection import train_test_split

# =============================================
tr = Trainer()
pp = Preprocessor()
# =============================================
print('#######################')
print('###    Welcome to   ###')
print('###      TE Gym     ###')
print('#######################\n')
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
                    help    = 'CSV file containing the hyperparametere metrics.')

parser.add_argument('-m',    '--metric',
                    default = 'val_loss',
                    help    = 'choose hyperparameters based on metric. Values are "val_loss" (default) or "val_accuracy".')

parser.add_argument('-t',    '--title',
                    default = 'TEGym',
                    type    = str,
                    help    = 'Model identifier (optional).')

parser.add_argument('-r',    '--runs',
                    default = 20,
                    type    = int,
                    help    = 'number of runs (tests) to find the hyperparameters.')

parser.add_argument('-s',    '--split',
                    default = 0.1,
                    type    = float,
                    help    = 'Portion of the dataset to use as validation set. The major portion is used for model training. Default=0.1')

args    = parser.parse_args()
# =============================================
# Variables
timestamp       = time.strftime('%y%m%d%H%M%S')
metric          = args.metric
sort_order      = True
hyper_option    = '--fasta'
model_title     = args.title
n_runs          = args.runs
n_split         = args.split
# =============================================
# Files
if args.fasta:
    infile = Path(args.fasta)
    fas_to_csv(infile)
elif args.csv:
    infile = Path(args.csv)
    hyper_option    = '--csv'
hyper_csv   = args.hyper
prefix      = f"{args.title}_{infile.stem}"
my_model    = Path(f"{prefix}_{timestamp}.keras")
hyper_out   = Path(f"hyperparams_{prefix}_in_{timestamp}.csv")
output_dir  = Path(f"{prefix}_{timestamp}")
out_config  = f"{prefix}_{timestamp}.toml"
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
else:
    subprocess.run(['python',       'hyperparameters.py',
                    hyper_option,   infile,
                    '--title',      model_title,
                    '--runs',       str(n_runs),
                    '--split',      str(n_split)])
# =============================================
if hyper_out.exists() or hyper_csv:
    # Preprocessing
    df  = pd.read_csv(f"{infile.stem}.csv",
                      usecols=['label', 'sequence']).sort_values('label')
    W   = pp.get_weight(df['label'])
    X   = pp.zero_padder(df['sequence'].map(pp.seq_tokenizer))
    y   = pp.transform_label(df['label'])
# =============================================
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify        = y,
                                                        test_size       = n_split,
                                                        random_state    = 13,
                                                        shuffle         = True)
# =============================================
    # Create model and save model
    print("### Starting training ###\n")
    model   = tr.create_model(hp_dict,
                              x_train.shape[1])
    fit     = tr.model_fit(model,
                           hp_dict,
                           x_train,
                           y_train,
                           x_test,
                           y_test,
                           W)

    model.save(my_model)
# =============================================
# =============================================
if my_model.exists():
    model_info = \
    f"""[model_info]
    model_name          = "{my_model}"
    Labels              = {df['label'].unique().tolist()}
    Maximum_length      = {x_train.shape[1]}
    Test_size           = {n_split}
    Training_samples    = {x_train.shape[0]}
    Validation_samples  = {x_test.shape[0]}
    """
    if not output_dir.exists():
        output_dir.mkdir()
    move(my_model, output_dir)
    with open(out_config, 'w') as sd:
        sd.write(model_info)
        move(out_config, output_dir)

print(">>> TEGym finished!")
