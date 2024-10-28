import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text

sample_data = [
    ('AI advancements in 2024', 'AI'),
    ('Understanding robotics technology', 'Robotics'),
]

df_sample = pd.DataFrame(sample_data, columns=['text', 'topic'])

df_sample['text'] = df_sample['text'].apply(preprocess_text)

label_encoder = LabelEncoder()
df_sample['encoded_topic'] = label_encoder.fit_transform(df_sample['topic'])

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english'),
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
)
model.fit(df_sample['text'], df_sample['encoded_topic'])

st.title("Article Topic Classification")

st.subheader("Single Article Classification")
text_input = st.text_area("Enter article text:", height=150)
if st.button("Classify"):
    if text_input:
        processed_text = preprocess_text(text_input)
        predicted_class = model.predict([processed_text])[0]
        predicted_topic = label_encoder.inverse_transform([predicted_class])[0]
        st.write(f"Predicted Topic: **{predicted_topic}**")
    else:
        st.warning("Please enter text to classify.")

st.subheader("Bulk Article Classification")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=['csv'])
if uploaded_file is not None:
    articles_df = pd.read_csv(uploaded_file)
    if 'text' in articles_df.columns:
        articles_df['clean_content'] = articles_df['text'].apply(preprocess_text)
        articles_df['predicted_topic'] = model.predict(articles_df['clean_content'])
        articles_df['predicted_topic'] = label_encoder.inverse_transform(articles_df['predicted_topic'])
        st.write("Classification Results")
        st.write(articles_df[['text', 'predicted_topic']])
        st.download_button("Download Results as CSV", articles_df.to_csv(index=False), "classified_articles.csv")
    else:
        st.error("Uploaded file must contain a 'text' column.")
