
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
with open("Intent.json", "r") as f:
    data = json.load(f)
texts = []
intents = []
for item in data["intents"]:
    intent_name = item["intent"]
    for example in item["text"]:
        texts.append(example)
        intents.append(intent_name)
df = pd.DataFrame({"text": texts, "intent": intents})
print(" Data extracted successfully")
print(df.head())
print("\nUnique Intents:", df['intent'].nunique())
print("Total Samples:", len(df))
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['intent']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nData split completed successfully")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
