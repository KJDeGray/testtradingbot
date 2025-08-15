import pandas as pd
import torch
import numpy as np
import time
from datetime import datetime


from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import Database as db
import Config as c

def normalize_matrix(df):
    return (df - df.min()) / (df.max() - df.min())



def Ticker_similarity_from_ticker_data(raw_ticker_data=None):


    # Load cached news articles
    if raw_ticker_data is None:
        print("Loading cached ticker data...")
        raw_ticker_data = db.load_cached_ticker()
        if not raw_ticker_data:
            print("No cached ticker data found.")
            return pd.DataFrame()  # Return empty DataFrame if no news available
    else:
        print(f"Using provided raw news data with {len(raw_ticker_data)} articles.")

    raw_ticker_data = db.load_cached_ticker()



    print(f"Loaded {len(raw_ticker_data)} tickers from cache.")
    cat_fields = ['exchange', 'sector', 'industry']
    raw_ticker_data = pd.DataFrame(raw_ticker_data)
    raw_ticker_data[cat_fields] = raw_ticker_data[cat_fields].fillna('')

    # Step 1: Encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    cat_features = encoder.fit_transform(raw_ticker_data[cat_fields])

    # Compute cosine similarity on categorical features
    S_cat = cosine_similarity(cat_features)

    # Step 2: Encode textual fields with Sentence-BERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Combine 'name' and 'description' for embedding (or do separately if preferred)
    texts = (raw_ticker_data['name'] + ". " + raw_ticker_data['description']).tolist()

    text_embeddings = model.encode(texts, convert_to_numpy=True)
    S_text = cosine_similarity(text_embeddings)

    # Step 3: Combine similarity matrices with weights
    #TODO: this is probably not the best way to combine these, but it works for now
    w_cat = 0.4
    w_text = 0.6

    S_combined = w_cat * S_cat + w_text * S_text

    return normalize_matrix(pd.DataFrame(S_combined, index=raw_ticker_data['symbol'], columns=raw_ticker_data['symbol']))


def Ticker_similarity_from_news(raw_news=None):
    """
    Calculate ticker similarity based on news articles.
    """
    # Load cached news articles
    if raw_news is None:
        print("Loading cached news articles...")
        raw_news = db.load_cached_news()
        if not raw_news:
            print("No cached news articles found.")
            return pd.DataFrame()  # Return empty DataFrame if no news available
    else:
        print(f"Using provided raw news data with {len(raw_news)} articles.")
    # Convert to DataFrame
    df = pd.DataFrame(raw_news)

    # Combine text fields into one column for embedding
    df['text'] = df['headline'].fillna('') + '. ' + df['summary'].fillna('') + '. ' + df['content'].fillna('') 

    # Optional: ensure symbols is always a list
    df['symbols'] = df['symbols'].apply(lambda x: x if isinstance(x, list) else [])

    # Keep only the important columns
    df = df[['date', 'author', 'source', 'text', 'symbols']]


    print(f"Processing {len(df)} news articles for ticker similarity. This may take a while...")
    start = datetime.now()
    # ======== Step 1: Embed articles ========
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, 384-dim embeddings
    article_embeddings = model.encode(df['text'].tolist(), convert_to_tensor=True)
    print(f"Article embeddings computed in {datetime.now() - start}")

    # ======== Step 2: Build ticker embeddings ========
    ticker_articles = defaultdict(list)

    print("Building ticker embeddings...")
    start = datetime.now()
    for idx, row in df.iterrows():
        for ticker in row['symbols']:
            ticker_articles[ticker].append(article_embeddings[idx])

    ticker_embeddings = {}
    for ticker, embeds in ticker_articles.items():
        if embeds:
            ticker_embeddings[ticker] = torch.mean(torch.stack(embeds), dim=0)
        else:
            ticker_embeddings[ticker] = None
    print(f"Ticker embeddings built in {datetime.now() - start}")


    # ======== Step 3: Calculate similarity (affiliation scores) ========
    all_tickers = sorted(ticker_embeddings.keys())
    embed_dim = next(v for v in ticker_embeddings.values() if v is not None).shape[0]
    ticker_matrix = torch.stack([
        ticker_embeddings[t] if ticker_embeddings[t] is not None else torch.zeros(embed_dim)
        for t in all_tickers
    ])
    # Normalize both article and ticker embeddings for cosine similarity
    article_norm = torch.nn.functional.normalize(article_embeddings, p=2, dim=1)  # [num_articles, embed_dim]
    ticker_norm = torch.nn.functional.normalize(ticker_matrix, p=2, dim=1)        # [num_tickers, embed_dim]

    # Cosine similarity = dot product between normalized vectors
    affiliation_scores = article_norm @ ticker_norm.T  # shape: [num_articles, num_tickers]

    # Convert to NumPy if needed
    affiliation_scores = affiliation_scores.cpu().numpy()
    # ======== Step 4: Put into DataFrame ========
    scores_df = pd.DataFrame(
        affiliation_scores,
        index=df.index,
        columns=all_tickers
    )

    ticker_similarity_from_news = pd.DataFrame(
        cosine_similarity(scores_df.T),
        index=all_tickers,
        columns=all_tickers
    )
    ticker_similarity_from_news = normalize_matrix(ticker_similarity_from_news)
    return ticker_similarity_from_news



def generate_combined_ticker_similarity(ticker_similarity_from_news=None, ticker_similarity_from_ticker_data=None, raw_ticker_data=None, raw_news_data=None):

    if raw_ticker_data is None:
        print("Loading cached ticker data...")
        raw_ticker_data = db.load_cached_ticker()
        if not raw_ticker_data:
            print("No cached ticker data found.")
            return pd.DataFrame()  # Return empty DataFrame if no ticker data available
    else:
        print(f"Using provided raw ticker data with {len(raw_ticker_data)} tickers.")
    
    if ticker_similarity_from_news is None:
        print("Calculating ticker similarity from news articles...")
        ticker_similarity_from_news = Ticker_similarity_from_news(raw_news=raw_news_data)
    else:
        print("Using provided ticker similarity from news articles.")
    if ticker_similarity_from_ticker_data is None:
        print("Calculating ticker similarity from ticker data...")
        ticker_similarity_from_ticker_data = Ticker_similarity_from_ticker_data(raw_ticker_data=raw_ticker_data)
    else:
        print("Using provided ticker similarity from ticker data.")

    if ticker_similarity_from_news.empty or ticker_similarity_from_ticker_data.empty:
        print("One of the ticker similarity matrices is empty. Cannot combine.")
        return pd.DataFrame()
    print("Combining ticker similarity matrices...")
    

    similarity_df = pd.DataFrame(
        ticker_similarity_from_ticker_data,
        index=raw_ticker_data['symbol'],
        columns=raw_ticker_data['symbol']
    )

    similarity_df = normalize_matrix(similarity_df)


    """"""""""""""""""""""""""""""""""""""""""""""""""""""

    w_meta = 0.8
    w_news = 0.2
    S_meta_norm = normalize_matrix(similarity_df)*w_meta
    print("S_meta_norm:", S_meta_norm)
    ticker_sim_norm = normalize_matrix(ticker_similarity_from_news)*w_news
    print("ticker_sim_norm:", ticker_sim_norm)

    snormsqare = S_meta_norm.multiply(S_meta_norm, fill_value=1)
    tickersqure = ticker_sim_norm.multiply(ticker_sim_norm, fill_value=1)


    combined_ticker_similarity = (snormsqare.add(tickersqure, fill_value=0)).add(ticker_sim_norm.multiply(S_meta_norm, fill_value=1), fill_value=0)
    combined_ticker_similarity = normalize_matrix(combined_ticker_similarity)

    print(combined_ticker_similarity)