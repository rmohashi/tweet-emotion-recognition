# Tweet Emotion Recognition

This repository contains the files to build a model to evaluate the emotions
from tweets.

## Setup

- Ensure you have the right versions of the dependencies. Run:

    ```bash
    pip install -r requirements.txt
    ```
- Create a `.env` file using the `.env.sample`:

    ```bash
    cp .env.sample .env
    ```
- Set the variables at the `.env` file

## Data Fetch

This module is responsible for the tweets gathering process.

### get_tweets

```bash
python -m data_fetch.get_tweets [QUERY] -s <SAVE_DIR> -m <MAX_REQUESTS> -c <COUNT> -n
```

Call the Twitter's API and search for tweets w/ the given `QUERY`.

#### Arguments

- **query**: String. Search parameter.
- **-s** | **--save_dir**: String. Path to the directory where the data will be saved.
- **-m** | **--max_requests**: Int. Max number of requests to the API. Default: `10`.
- **-c** | **--count**: Int. Max number of tweets for each request. Default: `100`.
- **-n** | **-newer**: Boolean. Collect the newer tweets. Default: `True`.

#### Returns

A `DataFrame` w/ the collected data when a `SAVE_DIR` is provided. Otherwise, `None`.

### concat_datasets

```bash
python -m data_fetch.concat_datasets [QUERY] -s <SAVE_DIR> -m <MAX_REQUESTS> -c <COUNT> -n
```

Create a new file w/ the datasets containing the given `QUERY`.

#### Arguments

- **query**: String. Search parameter.
- **dataset_dir**: String. Path to the directory where the files are saved.

### kmeans_data_filter

```bash
python -m data_fetch.kmeans_data_filter [FILE] -o <OUTPUT_NAME> -od <OUTPUT_DIR_NAME>
                                        -mf <MAX_FEATURES> -n <N_CLUSTERS>
                                        -ni <N_INIT> -mi <MAX_ITER>
                                        -r <RANDOM_STATE>
```

Filter the data from a file, based on the result of a clustering process using
the K-means algorithm.

#### Arguments

- **file**: String. Path to the file that will be processed.
- **-o** | **--output_name**: String. Name of the output file, including the
extension.
- **-od** | **--output_dir_name**: String. Name of the directory where the output
file will be placed. Default: `datasets/kmeans`.
- **-mf** | **--max_features**: Int. Max number of features to use for the TF-IDF
scoring. Default: `8000`.
- **-n** | **--n_clusters**: Int. Number of clusters to create. Default: `4`.
- **-ni** | **--n_init**: Int. Number of time the k-means algorithm will be run
with different centroid seeds. Default: `10`.
- **-mi** | **--max_iter**: Int. Maximum number of iterations of the k-means
algorithm for a single run. Default: `300`.
- **-r** | **--random_state**: Int. Determines random number generation for
centroid initialization. Default: `20`.

## Sentiment Analysis

This module is responsible for filtering the tweets, based the emotion associated w/
the query and the predicted sentiment.

### train

```bash
python -m sentiment_analysis.train [MODEL_TYPE] [DATASET_PATH] [TOKENIZER_PATH]
                                   [SAVE_DIR] -l <LABEL_COL> -t <TEXT_COL>
                                   -v <VALIDATION_SPLIT> -ed <EMBEDDING_DIM>
                                   -lr <LEARNING_RATE> -e <EPOCHS> -b <BATCH_SIZE>
```

Train a sentiment analysis model, using the given `MODEL_TYPE` and save the weights.

#### Arguments

- **model_type**: String. Model to use. Choices: [`lstm`, `lstm_conv`]
- **dataset_path**: String. Path to the dataset.
- **tokenizer_path**: String. Path to the tokenizer.
- **save_dir**: Stirng. Path to the directory where the weights will be saved.
- **-l** | **--label_col**: String. Name of the label column. Default: `label`.
- **-t** | **--text_col**: String. Name of the text column. Default: `text`.
- **-v** | **--validation_split**: Float. Fraction of the dataset to use as validation data. Default: `0.3`.
- **-ed** | **--embedding_dim**: Int. Output dimension of the embedding layer. Default: `100`.
- **-lr** | **--learning_rate**: Float. Initial learning rate. Default: `1e-3`.
- **-e** | **--epochs**: Int. Total number of epochs. Default: `10`.
- **-b** | **--batch_size**: Int. Number of batches per epoch. Default: `32`.

### train_nb

```bash
python -m sentiment_analysis.train_nb [DATA_FILE] [MODEL_FILE] [SAVE_DIR] -t <TEXT_COL> -p
```

Train a naive bayes model and save it.

#### Arguments

- **filename**: String. Path to the dataset.
- **-l** | **label_col**: String: Name of the label column. Default: `label`.
- **-t** | **text_col**: String: Name of the text column. Default: `text`.
- **-t** | **validation_split**: Float. Fraction of the dataset to use as validation data. Default: `0.3`.

### predict_from_directory

```bash
python -m sentiment_analysis.predict_from_directory [FILES_DIR] [MODEL_FILE] [SAVE_PATH] -t <TEXT_COL>
```

Create an annotated dataset, based on the search query and the Naive Bayes' sentiment analysis.

#### Arguments

- **files_dir**: String. Path to the directory containing the datasets.
- **model_file**: String. Path to the naive bayes `.pickle` file.
- **save_path**: String. Path where the resulting dataset will be saved.
- **-t** | **--text_col**: String. Name of the text column. Default: `text`.

### predict_nb

```bash
python -m sentiment_analysis.predict_nb [DATA_FILE] [MODEL_FILE] [SAVE_DIR] -t <TEXT_COL> -p
```

Predict the tweet polarity and save the purge the oddly fetched examples.

#### Arguments

- **data_file**: String. Path to the data file.
- **model_file**: String. Path to the model file.
- **save_dir**: String. Path to the directory where the result will be saved.
- **-t** | **--text_col**: String. Name of the text column. Default: `text`.
- **-p** | **--positive**: Boolean. If the emotion can be classified as positive. Default: `False`.
