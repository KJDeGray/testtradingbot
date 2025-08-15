import os
import Secrets


API_KEY = Secrets.API_KEY
API_SECRET = Secrets.API_SECRET

DATA_BASE_URL = "https://data.alpaca.markets"
NEWS_BASE_URL = "https://data.alpaca.markets/v1beta1/news"
TRADING_BASE_URL = "https://paper-api.alpaca.markets"

BACKTESTING_START_DATE = "2023-01-01"
BACKTESTING_END_DATE = "2023-01-05"

#folder paths
MODEL_DIR = os.path.join(os.getcwd(), "model_storage")
DATA_DIR = os.path.join(os.getcwd(), "data_storage")

# Ensure folder exists
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

#saved names for files
RAW_NEWS_NAME = "news_data"
RAW_TICKER_NAME = "ticker_data"
RAW_BARS_NAME = "bars_data"
AFFILIATION_NAME = "affiliation_data"

API_SS = False #set to false if you want to use the API to download data
DAY_ONLY = True #set to false if you want minute data

# Function to create file paths for models and data
def make_model_filepath(name, suffix = ".pkl", start_date=BACKTESTING_START_DATE, end_date=BACKTESTING_END_DATE):
    return os.path.join(MODEL_DIR, name +start_date+end_date+ suffix)

def make_data_filepath(name, suffix = ".json",start_date=BACKTESTING_START_DATE, end_date=BACKTESTING_END_DATE):
    return os.path.join(DATA_DIR, name+start_date+end_date+suffix)

#RAW_BARS_FILE = os.path.join(DATA_DIR, "bars_data"+BACKTESTING_START_DATE+BACKTESTING_END_DATE+".json")
#RAW_NEWS_FILE = os.path.join(DATA_DIR, "news_data"+BACKTESTING_START_DATE+BACKTESTING_END_DATE+".json")
#RAW_TICKER_FILE = os.path.join(DATA_DIR, "ticker_data"+BACKTESTING_START_DATE+BACKTESTING_END_DATE+".json")

