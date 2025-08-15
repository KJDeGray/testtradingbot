import importlib
import yfinance as yf
import pandas as pd
import time
import json
import Database as db
import Config as c
importlib.reload(db)
importlib.reload(c)




def retrieve_ticker_data(tickers):
        data = []
        symbol_number = 0
        for symbol in tickers:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                data.append({
                    "symbol": symbol,
                    "name": info.get("longName"),
                    "exchange": info.get("exchange"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "description": info.get("longBusinessSummary")
                })
                symbol_number += 1
                print(f"Retrieved symbol {symbol} at number: {symbol_number}                                                                                  ",end='\r', flush=True)
                time.sleep(0.2)  # small delay to avoid rate limits
            except Exception as e:
                print(f"Error for {symbol}: {e} on ticker {symbol}                                                                                            ")
        return data


def raw_ticker_retriever(save = False):
    # Load cached news articles
    news = db.load_cached_news()
    tickers = sorted({ticker for article in news for ticker in article["symbols"]})
    if c.API_SS:
        print("SS is on, ticker data will not be downloaded")
        return None
    else:
        print("\n\nWARNING!!!\n SS IS OFF, TICKER DATA WILL BE DOWNLOADED\n\n")
        print(f"Total tickers to retrieve: {len(tickers)}")
        
        # Retrieve ticker data
        data = retrieve_ticker_data(tickers)
        if save:
            db.save_cached_ticker(data)
            print(f"Saved {len(data)} ticker entries to {c.make_data_filepath(c.RAW_TICKER_NAME)}")
        return data



if __name__ == "__main__":
    data = raw_ticker_retriever(save = True)