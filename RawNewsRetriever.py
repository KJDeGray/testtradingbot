import requests
import time
from datetime import datetime

import Config as c
import Database as db




def get_news_from_alpaca(symbol=None, query_start=datetime.strptime(c.BACKTESTING_START_DATE, '%Y-%m-%d').date(), query_end=datetime.strptime(c.BACKTESTING_END_DATE, '%Y-%m-%d').date(), base_url = c.NEWS_BASE_URL,headers=None):

    params = {
        "symbols": symbol,
        "start": query_start,
        "end": query_end,
        "limit": 50,
        "sort": "desc"
    }
    print(f"Fetching news from Alpaca API for symbols: {symbol} from {query_start} to {query_end}")

    page_token = None
    all_news = []
    page_number = 0
    total_items = 0
    while True:
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)

        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code == 429:
            print("Rate limited (429). Waiting 60 seconds...")
            time.sleep(60)
            continue
        elif response.status_code != 200:
            raise Exception(f"Alpaca API error: {response.status_code}, {response.text} on ticker {symbol}                                                                                        ")
        else:
            page_number += 1
            size = len(response.json().get("news", []))
            total_items += size
            print(f"Fetched news page {page_number} with {size} items totaling {total_items} items",end='\r', flush=True)
            

        data = response.json()
        news_page = data.get("news", [])
        all_news.extend(news_page)

        page_token = data.get("next_page_token")
        if not page_token:
            break

        time.sleep(0.1)

    return all_news



def clean_news_data(news, remove_minutes=c.DAY_ONLY):

    stripped_news = []
    for article in news:
        new_article = article.copy()
        if "id" in new_article:
            new_article.pop("id")
        if "updated_at" in new_article:
            new_article.pop("updated_at")
        if "created_at" in new_article:
            new_article["date"] = datetime.fromisoformat(new_article.pop("created_at").replace("Z", "+00:00"))
            if remove_minutes:
                new_article["date"] = new_article["date"].date()
        if "images" in new_article:
            new_article.pop("images")
        if "url" in new_article: 
            new_article.pop("url")
        stripped_news.append(new_article)

    return stripped_news

def raw_news_retriever(save = False):
    if c.API_SS:
        print("SS is on, news data will not be downloaded")
        return None
    else:
        print("\n\nWARNING!!!\n SS IS OFF, NEWS WILL BE DOWNLOADED\n\n")
        headers = {
            "APCA-API-KEY-ID": c.API_KEY,
            "APCA-API-SECRET-KEY": c.API_SECRET,
        }
        news = get_news_from_alpaca(headers=headers)
        cleaned_news = clean_news_data(news)
        if save:
            db.save_cached_news(cleaned_news)
            print(f"Saved {len(cleaned_news)} news articles to {c.make_data_filepath(c.RAW_NEWS_NAME)}")
        else:
            print(f"Retrieved {len(cleaned_news)} news articles without saving.")
        return cleaned_news

if __name__ == "__main__":
    raw_news_retriever(save = True)
