
import streamlit as st
import pandas as pd
import re
import spacy
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Symptom2Disease_cleaned_final.csv")
    df.dropna(subset=["text", "label", "treatment"], inplace=True)
    return df

df = load_data()

# Build dynamic treatment lookup
treatment_lookup = df.groupby("label")["treatment"].first().to_dict()

# Advanced Preprocessing
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

df['clean_symptoms'] = df['text'].apply(preprocess)

# TF-IDF + Model Training with Logistic Regression
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_symptoms'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(lr_model, "disease_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Evaluate model
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit Interface
st.title("🧠 Disease Prediction from Symptoms")

user_input = st.text_area("Enter your symptoms (comma-separated):")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter at least one symptom.")
    else:
        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = lr_model.predict(vector)[0]
        proba = lr_model.predict_proba(vector)[0]
        treatment = treatment_lookup.get(prediction, "No safety measures or treatment found.")
        st.success(f"🦠 Predicted Disease: {prediction}")
        st.info(f"💊 Suggested Treatment or Safety Measures\n{treatment}")
