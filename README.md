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
python -m data_fetch [QUERY] -s <SAVE_DIR> -m <MAX_REQUESTS> -c <COUNT> -n
```

#### Arguments

- **query**: String. Search parameter.
- **-s** | **--save_dir**: String. Path to the directory where the data will be saved.
- **-m** | **--max_requests**: Int. Max number of requests to the API.
- **-c** | **--count**: Int. Max number of tweets for each request.
- **-n** | **-newer**: Boolean.

#### Returns

A `DataFrame` w/ the collected data when a `SAVE_DIR` is provided. Otherwise, `None`
