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


## Sentiment Analysis

This module is responsible for filtering the tweets, based the emotion associated w/
the query and the predicted sentiment.

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

## Emotion Recognition

### create_dataset

```bash
python -m emotion_recogntion.create_dataset [FILES_DIR] [SAVE_DIR]
```

Create an annotated dataset, based on the search query.

#### Arguments

- **files_dir**: String. Path to the folder with the Twitter data.
- **save_dir**: String. Path to the directory where the result will be saved.
