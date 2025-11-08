#Week 1
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

#Week 2

# Model Training, Evaluation & Metric Visualization


import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ✅ Create a folder to store metric graphs
os.makedirs("results", exist_ok=True)

# Define models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(random_state=42)
}

results = {}

print("\n================ MODEL TRAINING & EVALUATION ================\n")

for model_name, model in models.items():
    print(f"🔹 Training {model_name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = acc

    print(f"\n✅ {model_name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save confusion matrix as image
    file_path = f"results/{model_name.replace(' ', '_')}_ConfusionMatrix.png"
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"📊 Confusion matrix saved to: {file_path}")
    print("-------------------------------------------------------------\n")

# Compare model accuracies visually
plt.figure(figsize=(8, 6))
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()

# Save accuracy comparison chart
comparison_path = "results/Model_Accuracy_Comparison.png"
plt.savefig(comparison_path, dpi=300)
plt.close()

print("🎯 Model Comparison Results:")
for model_name, acc in results.items():
    print(f"{model_name:25s}: {acc:.4f}")

best_model_name = max(results, key=results.get)
print(f"\n🏆 Best Performing Model: {best_model_name} ({results[best_model_name]:.4f} accuracy)")

print(f"\n✅ All metric graphs saved successfully in the 'results/' folder.")


