import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 65)
print(" EXPERIMENT 05: NAIVE BAYES TEXT CLASSIFICATION DEMO")
print("=" * 65)

# 1. Load Dataset
data_path = os.path.join(os.path.dirname(__file__), "datasets", "movie_data.csv.gz")
print(f"\n[STEP 1] Loading IMDB Dataset from: {data_path} ...")
df = pd.read_csv(data_path, compression="gzip", nrows=2500)
print(f" -> Total Loaded Documents: {df.shape[0]} reviews")
print(f" -> Sentiment Distribution:\n{df['sentiment'].value_counts().to_string()}")

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, stratify=df["sentiment"], random_state=42
)
print(f"\n[STEP 2] Train-Test Split Completed:")
print(f" -> Training set size : {len(X_train)} reviews")
print(f" -> Testing set size  : {len(X_test)} reviews")

# 3. Build TF-IDF + Multinomial Naive Bayes Pipeline
print(f"\n[STEP 3] Training Pipeline (TfidfVectorizer -> MultinomialNB)...")
pipeline = make_pipeline(
    TfidfVectorizer(stop_words="english", max_features=5000, lowercase=True),
    MultinomialNB(alpha=1.0)
)
pipeline.fit(X_train, y_train)
print(" -> Pipeline trained successfully!")
print(f" -> Extracted Vocabulary Features: {len(pipeline[0].vocabulary_)} top n-grams")

# 4. Evaluate on Test Set
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)
cm = confusion_matrix(y_test, y_pred)

print(f"\n[STEP 4] Evaluation Results:")
print(f" -> Model Test Accuracy: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print("                 Predicted Negative  Predicted Positive")
print(f"Actual Negative        {cm[0, 0]:<18}  {cm[0, 1]:<18}")
print(f"Actual Positive        {cm[1, 0]:<18}  {cm[1, 1]:<18}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"], digits=4))

# 5. Live Demonstration with Custom Reviews
print("=" * 65)
print("[STEP 5] LIVE INFERENCE DEMONSTRATION")
print("=" * 65)
demo_reviews = [
    "An absolute masterpiece! The acting was brilliant and the storyline kept me engaged till the very end.",
    "A total waste of time, completely terrible screenplay and wooden acting.",
    "Visually stunning cinematography paired with a deeply emotional musical score. Highly recommended!",
    "Boring, dull, and predictable. One of the worst films I have ever watched in a cinema.",
    "The movie was okay, had some decent comedy scenes but the ending felt rushed."
]

predictions = pipeline.predict(demo_reviews)
probabilities = pipeline.predict_proba(demo_reviews)

for i, (rev, pred, prob) in enumerate(zip(demo_reviews, predictions, probabilities), 1):
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = max(prob) * 100
    print(f"\nTest Review #{i}: \"{rev}\"")
    print(f" -> PREDICTION : [{label}] (Confidence: {confidence:.2f}%)")

print("\n" + "=" * 65)
print("Demo completed successfully!")
print("=" * 65)
