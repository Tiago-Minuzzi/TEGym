# TEGym

## What is TEGym?

TEGym is a program written in python to help people without deep learning expertise create a data driven transposable elements classifier, i.e., a model to classify transposable elements of species lacking enough data to train a classifier, using the data from a more
closely related species. It automatizes preprocessing, hyperparameters testing and model training, resulting in a classifier suited for the needs of the user.

For a better explanation on how to use the program, check the manual in PDF.

TEGym is a work in progress to the date of this writing. We are trying to add more options and improvements as soon as possible.

## Basic usage

The most basic usage is simple. You only need a `FASTA` file or a `CSV` file contaning the sequences and the labels. The `CSV` table must contain the columns named `label` and `sequences`. The `FASTA` header/id must be in the `RepeatMasker` format (`sequenceID#label`).

Example:

`python gym.py -f my_file.fasta`

or

`python gym.py -c my_file.csv`

The initial phase involves searching for the optimal hyperparameters to train the model based on the input dataset. Then, the model will be trained using the best combination of hyperparameters, determined by the lowest validation loss.

## Running steps independently

Instead of running hyperparameter search and model training all at once, you can run the steps independentely. Just call the script hyperparameters.py to generate the CSV to be used for model training later. Then, when calling `gym.py` use the flag `-p` to indicate the path to the hyperparameter’s CSV.

Example:

	python hyperparameter.py -f my_file.fasta
	python gym.py -f my_file.fasta -p TEGym_hyperparameters.csv

## Customizing hyperparameters search
If you want to set-up other values for hyperparameter searching different than the default values used by TEGym, you just need to modify the values in the TOML file my_config_hyperparameters.toml.

Do NOT change the name of values before the = sign, just the values inside square brackets, which must be comma separated.

## Other flags

You can view other flags and their usage by running:

`python gym.py --help`

or

`python hyperparameters.py --help`

	usage: gym.py [-h] (-f FASTA | -c CSV) [-p HYPER] [-m METRIC] [-t TITLE] [-r RUNS] [-s SPLIT]
	
	Train your own classifier model.
	
	options:
	  -h, --help            show this help message and exit
	  -f FASTA, --fasta FASTA
	                        Input fasta file with id and labels formatted as: ">seqId#Label".
	  -c CSV, --csv CSV     Input CSV file containing columns "id", "label", "sequence".
	  -p HYPER, --hyper HYPER
	                        CSV file containing the hyperparametere metrics.
	  -m METRIC, --metric METRIC
	                        choose hyperparameters based on metric. Values are "val_loss" (default) or "val_accuracy".
	  -t TITLE, --title TITLE
	                        Model identifier (optional).
	  -r RUNS, --runs RUNS  number of runs (tests) to find the hyperparameters.
	  -s SPLIT, --split SPLIT
	                        Portion of the dataset to use as validation set. The major portion is used for model training. Default=0.1.

## FASTA to CSV

When using a FASTA file as input, the program will convert it to a CSV file. Depending on the size of your FASTA, it may be time-consuming. You can convert you FASTA to CSV prior to running the program using the script fasta_to_csv.py as follows:

`python fasta_to_csv.py my_file.fasta`

---