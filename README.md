# TEGym

Build your own deep learning transposable elements classifier with ease.

## Table of contents

- [What is TEGym?](#Whatis)

- [Installation](#install)

- [Basic usage](#usage)

- [Running steps independentely](#running)

- [Customizing hyperparameters search](#custom)

- [FASTA to CSV](#ftc)
    
- [Predict](#predict)

- [Create negative class](#negative)

## What is TEGym? <a name="Whatis"></a>

TEGym is a program written in python to help people without deep learning expertise create a data driven transposable elements classifier, i.e., a model to classify transposable elements of species lacking enough data to train a classifier, using the data from a more
closely related species. It automatizes preprocessing, hyperparameters testing and model training, resulting in a classifier suited for the needs of the user. Although TEGym was developed with transposable elements in mind, it can probably be used for other sequence classification tasks.

For a better explanation on how to use the program, check the manual in PDF.

TEGym is a work in progress to the date of this writing. We are trying to add more options and improvements as soon as possible.

## Installation <a name="install"></a>

TEGym uses python version 3.11. Preferentially use python version >= 3.10 in a python virtual enviroment or a conda enviroment. Install the required packages using:

`pip install -r requirements.txt`

## Basic usage <a name="usage"></a>

The most basic usage is simple. You only need a `FASTA` file or a `CSV` file contaning the sequences and the labels. The `CSV` table must contain the columns named `label` and `sequences`. The `FASTA` header/id must be in the `RepeatMasker` format (`sequenceID#label`).

Example:

`python gym.py -f my_file.fasta`

or

`python gym.py -c my_file.csv`

The initial phase involves searching for the optimal hyperparameters to train the model based on the input dataset. Then, the model will be trained using the best combination of hyperparameters, determined by the lowest validation loss.

## Running steps independently <a name="running"></a>

Instead of running hyperparameter search and model training all at once, you can run the steps independently. Just call the script hyperparameters.py to generate the CSV to be used for model training later. Then, when calling `gym.py` use the flag `-p` to indicate the path to the hyperparameter’s CSV.

Example:

	python hyperparameter.py -f my_file.fasta
	python gym.py -f my_file.fasta -p TEGym_hyperparameters.csv

## Customizing hyperparameters search <a name="custom"></a>
If you want to set-up other values for hyperparameter searching different than the default values used by TEGym, you just need to modify the values in the TOML file my_config_hyperparameters.toml.

Do NOT change the name of values before the = sign, just the values inside square brackets, which must be comma separated.

## Other flags <a name="other"></a>

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

## FASTA to CSV <a name="ftc"></a>

When using a FASTA file as input, the program will convert it to a CSV file. Depending on the size of your FASTA, it may be time-consuming. You can convert you FASTA to CSV prior to running the program using the script fasta_to_csv.py as follows:

`python fasta_to_csv.py my_file.fasta`

## Predict <a name="predict"></a>

After your model is trained using gym.py, you can use it as a classifier by running the script predict.py. It has three mandatory arguments: an FASTA file with sequences to be classified, the path to the trained model and the path to the TOML file with model info.

`python predict.py -f file.fasta -m my_model.keras -i my_model_info.toml`

The output is a CSV file containing the classification prediction for
each sequence and the classication score ranging from 0 to 1.

| id | prediction | TE_score | NonTE_score |
| :---: | :---: | :---: | :---: |
| Seq01 | TE | 0.98 | 0.02 |
| Seq02 | TE | 0.72 | 0.28 |
| Seq03 | NonTE | 1.0 | 0.0 |
| Seq04 | NonTE | 0.63 | 0.37 |
| Seq05 | TE | 0.85 | 0.15 |

## Create a negative class <a name="negative"></a>

If your dataset has only one class, for instance, only sequences labeled as `TE`, you can use the script `create_negative_class.py` to create another class to train your model. Use the values `random` or `shuffled` with the flag `-c` to create random sequences or shuffle your sequences, respectively.

Example:

`python create_negative_class.py -f my_file.fasta -c shuffled`.

The output is a `CSV` file with the prefix `TDS` containing your sequences and the newly created ones. Then you can use it with the main program.


---

## TO-DO

- [x] Generate random sequences if only one is class available.
- [ ] Option to generate reverse complement.
- [ ] Add example files.
- [ ] Option to use k-mers.
- [ ] Add GPU support.
