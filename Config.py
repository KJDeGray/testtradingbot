import os

API_KEY = "PKJUP996961KC63A7FWO" 
API_SECRET = "FlS4JsxyehoBzODmBHBofbPyaWuqgq3QY7XyIFer" 

DATA_BASE_URL = "https://data.alpaca.markets"
TRADING_BASE_URL = "https://paper-api.alpaca.markets"

BACKTESTING_START_DATE = "2023-01-01"
BACKTESTING_END_DATE = "2023-04-01"

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




# Function to create file paths for models and data
def make_model_filepath(name, suffix = ".pkl", start_date=BACKTESTING_START_DATE, end_date=BACKTESTING_END_DATE):
    return os.path.join(MODEL_DIR, name +start_date+end_date+ suffix)

def make_data_filepath(name, suffix = ".json",start_date=BACKTESTING_START_DATE, end_date=BACKTESTING_END_DATE):
    return os.path.join(DATA_DIR, name+start_date+end_date+suffix)

#RAW_BARS_FILE = os.path.join(DATA_DIR, "bars_data"+BACKTESTING_START_DATE+BACKTESTING_END_DATE+".json")
#RAW_NEWS_FILE = os.path.join(DATA_DIR, "news_data"+BACKTESTING_START_DATE+BACKTESTING_END_DATE+".json")
#RAW_TICKER_FILE = os.path.join(DATA_DIR, "ticker_data"+BACKTESTING_START_DATE+BACKTESTING_END_DATE+".json")

