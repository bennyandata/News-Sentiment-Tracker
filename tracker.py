from config import KEYWORDS
from sentiment_analyze import *
from datetime import datetime, timedelta
import pandas as pd
import os

# today = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
today = "2025-07-25"
results = []

for kw in KEYWORDS:
    result = analyze_sentiment(kw, today)
    result["date"] = today
    results.append(result)

df = pd.DataFrame(results)

os.makedirs("data", exist_ok=True)
csv_path = "data/sentiment_log.csv"

if os.path.exists(csv_path):
    df.to_csv(csv_path, index=False, mode="a", header=False)
else:
    df.to_csv(csv_path, index=False)

print(df)
