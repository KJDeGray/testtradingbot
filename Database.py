import requests
import time
import pandas as pd
import dill
import os
import json
from datetime import date, timedelta, datetime
import Config as c

def save_cached_news(cache):
    for element in cache:
        if isinstance(element, dict):
            for key, value in element.items():
                if isinstance(value, datetime):
                    element[key] = value.isoformat()
                elif isinstance(value, date):
                    element[key] = value.isoformat()
        else:
            raise ValueError("Cache must be a list of dictionaries with datetime or date values.")
    with open(c.make_data_filepath(c.RAW_NEWS_NAME), 'w') as f:
        json.dump(cache, f, indent=4, default=str)


def load_cached_news():
    if os.path.exists(c.make_data_filepath(c.RAW_NEWS_NAME)):
        with open(c.make_data_filepath(c.RAW_NEWS_NAME), 'r') as f:
            raw = json.load(f)
        for element in raw:
            for key, value in element.items():
                if isinstance(value, str):
                    try:
                        element[key] = datetime.fromisoformat(value)
                    except ValueError:
                        try:
                            element[key] = date.fromisoformat(value)
                        except ValueError:
                            pass
        return raw
    else:     
        print(f"Cache file {c.make_data_filepath(c.RAW_NEWS_NAME)} does not exist.")
    return {}

def save_cached_ticker(cache):
    with open(c.make_data_filepath(c.RAW_TICKER_NAME), 'w') as f:
        json.dump(cache, f, indent=4, default=str)

def load_cached_ticker():
    if os.path.exists(c.make_data_filepath(c.RAW_TICKER_NAME)):
        with open(c.make_data_filepath(c.RAW_TICKER_NAME), 'r') as f:
            raw = json.load(f)
        return raw
    else:     
        print(f"Cache file {c.make_data_filepath(c.RAW_TICKER_NAME)} does not exist.")
    return {}


def save_affiliation_data(state):
    with open(c.make_data_filepath(c.AFFILIATION_NAME,suffix=".pkl"), 'wb') as f:
        dill.dump(state, f)

def load_affiliation_data(base_filename = c.DATA_DIR):
    with open(c.make_data_filepath(c.AFFILIATION_NAME,suffix=".pkl"), 'rb') as f:
        state = dill.load(f)
    return state

