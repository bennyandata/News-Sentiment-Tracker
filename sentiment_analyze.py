from dotenv import load_dotenv
import os
import requests
import torch
from transformers import pipeline
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("API_KEY")

pipe = pipeline("text-classification", model="ProsusAI/finbert")


def fetch_articles(keyword, date):
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={keyword}&"
        f"from={date}&"
        "sortBy=popularity&"
        f"apiKey={API_KEY}"
    )
    response = requests.get(url)

    print(response.status_code, response.url)
    print(response.json())

    return response.json().get("articles", [])


# def filter_valid_articles(articles, keyword):
#     # articles = [article for article in articles if keyword.lower() in article['title'].lower() or keyword.lower() in article['description'].lower()]
#     # return articles

#     return [
#         article
#         for article in articles
#         if (article.get("title") and keyword.lower() in article["title"].lower())
#         or (
#             article.get("description")
#             and keyword.lower() in article["description"].lower()
#         )
#     ]


def filter_valid_articles(articles, keyword):
    keywords = [k.strip().lower() for k in keyword.split("OR")]

    return [
        article
        for article in articles
        if any(
            kw
            in (
                (article.get("title") or "").lower()
                + (article.get("description") or "").lower()
            )
            for kw in keywords
        )
    ]


def analyze_sentiment(keyword, date):
    articles = fetch_articles(keyword, date)
    articles = filter_valid_articles(articles, keyword)

    if not articles:
        return {
            "keyword": keyword,
            "score": 0,
            "sentiment": "No Articles",
            "num_articles": 0,
        }

    total_score = 0
    num_articles = 0

    for i, article in enumerate(articles):
        print(f"Title:{article['title']}")
        print(f"Link:{article['url']}")
        print(f"Description:{article['description']}")

        sentiment = pipe(article["content"])[0]
        print(f"Sentiment {sentiment['label']},Score:{sentiment['score']}")
        print("-" * 40)

        if sentiment["label"] == "positive":
            total_score += sentiment["score"]
            num_articles += 1
        elif sentiment["label"] == "negative":
            total_score -= sentiment["score"]
            num_articles += 1

    # if num_articles == 0:
    #     return {"keyword": keyword, "score": 0, "sentiment": "No Valid Content", "num_articles": 0}

    avg_score = total_score / num_articles
    sentiment = (
        "Positive"
        if avg_score > 0.15
        else "Negative"
        if avg_score < -0.15
        else "Neutral"
    )
    return {
        "keyword": keyword,
        "score": round(avg_score, 4),
        "sentiment": sentiment,
        "num_articles": num_articles,
    }
