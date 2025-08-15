import torch
import pandas as pd
from sentence_transformers import SentenceTransformer


import Database as db
import Config as c
import os


def build_affiliation_state(df,model = None,save = True):

    # Ensure symbols list
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    df['symbols'] = df['symbols'].apply(lambda x: x if isinstance(x, list) else [])

    texts = (df['headline'].fillna('') + '. ' + df['summary'].fillna('')).tolist()
    article_embeddings = model.encode(texts, convert_to_tensor=True)

    # Build ticker index map
    all_tickers = sorted({ticker for symbols in df['symbols'] for ticker in symbols})
    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}

    embed_dim = article_embeddings.shape[1]
    ticker_sums = torch.zeros(len(all_tickers), embed_dim, device=article_embeddings.device)
    ticker_counts = torch.zeros(len(all_tickers), device=article_embeddings.device)

    for i, tickers in enumerate(df['symbols']):
        emb = article_embeddings[i]
        for ticker in tickers:
            idx = ticker_to_idx[ticker]
            ticker_sums[idx] += emb
            ticker_counts[idx] += 1
        print(f"Processed {i+1}/{len(df)} articles", end='\r', flush=True)

    ticker_embeddings = {t: ticker_sums[i] / ticker_counts[i] for t, i in ticker_to_idx.items()}

    state = {
        "ticker_sums": {t: ticker_sums[i] for t, i in ticker_to_idx.items()},
        "ticker_counts": {t: int(ticker_counts[i].item()) for t, i in ticker_to_idx.items()},
        "ticker_embeddings": ticker_embeddings,
        "all_tickers": all_tickers
    }

    if save:
        db.save_affiliation_data(state)
    return state


def add_article(article, state,model=None):
    """
    Add a single article to existing scores_df and update state.
    
    article: dict with 'headline', 'summary', 'symbols' (list)
    state: dict returned from build_affiliation_scores
    """

    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    ticker_sums = state["ticker_sums"]
    ticker_counts = state["ticker_counts"]
    ticker_embeddings = state["ticker_embeddings"]
    all_tickers = state["all_tickers"]

    # Embed new article
    text = article['headline'] + '. ' + article['summary']
    embed = model.encode(text, convert_to_tensor=True)

    # Handle new tickers
    for ticker in article['symbols']:
        if ticker not in ticker_sums:
            ticker_sums[ticker] = torch.zeros(embed.shape)
            ticker_counts[ticker] = 0
            ticker_embeddings[ticker] = torch.zeros(embed.shape)
            all_tickers.append(ticker)

        ticker_sums[ticker] += embed
        ticker_counts[ticker] += 1
        ticker_embeddings[ticker] = ticker_sums[ticker] / ticker_counts[ticker]

    # Compute new similarity row
    row_scores = []
    for ticker in all_tickers:
        if ticker_counts[ticker] > 0:
            score = torch.nn.functional.cosine_similarity(
                embed.unsqueeze(0),
                ticker_embeddings[ticker].unsqueeze(0)
            ).item()
        else:
            score = 0.0
        row_scores.append(score)

    # Append new row
    new_row = pd.DataFrame([row_scores], columns=all_tickers)

    # Save updated state
    state["ticker_sums"] = ticker_sums
    state["ticker_counts"] = ticker_counts
    state["ticker_embeddings"] = ticker_embeddings
    state["all_tickers"] = all_tickers

    return new_row, state


if __name__ == "__main__":
    if not os.path.exists(c.make_data_filepath(c.RAW_NEWS_NAME)):
        print(f"Cache file {c.make_data_filepath(c.RAW_NEWS_NAME)} does not exist. Please run news collection first.")
        exit(1)
    else:
        
        df = pd.Dataframe(c.load_cached_news())
        print("building affiliation state. This may take a while...")
        affiliation_state = build_affiliation_state(df)
        print("Affiliation state built successfully.")


    
    


    


