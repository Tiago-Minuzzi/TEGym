from neural_net import Trainer
from preprocessing import Preprocessor

# =============================================

tr = Trainer()
pp = Preprocessor()

# =============================================

# Helper
parser  = argparse.ArgumentParser(prog          = 'hyperparameters.py',
                                  description   = "Search hyperparametere
s for model training.")

group   = parser.add_mutually_exclusive_group(required = True)

group.add_argument('-f',    '--fasta',
                    help    = '''Input fasta file with id and labels formatted as: ">seqId#Label".''')

group.add_argument('-c',    '--csv',
                    help    = '''Input CSV file containing columns "id",
"label", "sequence".''')

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
