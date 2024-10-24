import pandas as pd
!pip install nltk
import nltk
nltk.download('wordnet')
import re
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk


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
    ('How waste management can save the planet', 'Waste Management'),
    ('Latest trends in technology', 'Technology'),
    ('Revolution in renewable energy', 'Sustainability'),
    ('The future of artificial intelligence', 'AI'),
    ('Ethics in robotics development', 'Robotics'),
    ('The importance of recycling', 'Waste Management'),
    ('Emerging technologies in IT', 'Technology'),
    ('Sustainable practices for businesses', 'Sustainability'),
    ('Machine learning applications in healthcare', 'AI'),
    ('Autonomous vehicles and their impact', 'Robotics'),
    ('Waste reduction strategies', 'Waste Management'),
    ('Innovative tech solutions for climate change', 'Sustainability'),
    ('Trends in artificial intelligence research', 'AI'),
    ('Robotic process automation in industries', 'Robotics'),
    ('AI in healthcare: transforming patient outcomes', 'AI'),
    ('Robotics in agriculture', 'Robotics'),
    ('Plastic waste management strategies', 'Waste Management'),
    ('Technology trends for 2025', 'Technology'),
    ('Corporate sustainability initiatives', 'Sustainability'),
    ('AI and its role in future industries', 'AI'),
    ('Latest innovations in robotics', 'Robotics'),
    ('Strategies for effective waste management', 'Waste Management'),
    ('Impact of technology on society', 'Technology'),
    ('Global sustainability goals', 'Sustainability'),
    ('Deep learning breakthroughs in AI', 'AI'),
    ('AI ethics and accountability', 'AI'),
    ('Smart robotics in modern manufacturing', 'Robotics'),
    ('Composting methods for urban areas', 'Waste Management'),
    ('Green technology initiatives', 'Sustainability'),
    ('Natural language processing applications', 'AI'),
    ('Robotics in healthcare: improving patient care', 'Robotics'),
    ('Circular economy and waste management', 'Waste Management'),
    ('Technological advancements in renewable energy', 'Sustainability'),
    ('AI for data-driven decision making', 'AI'),
    ('Drones and robotics in disaster management', 'Robotics'),
    ('Waste recycling technologies', 'Waste Management'),
    ('Digital transformation in businesses', 'Technology'),
    ('Sustainable agriculture practices', 'Sustainability'),
    ('AI in financial services', 'AI'),
    ('Humanoid robots and their future', 'Robotics'),
    ('Policy changes in waste management', 'Waste Management'),
    ('The rise of 5G technology', 'Technology'),
    ('Renewable energy sources and their impact', 'Sustainability'),
    ('AI in education: personalized learning', 'AI'),
    ('The impact of AI on jobs', 'AI'),
    ('Collaborative robots: the future of work', 'Robotics'),
    ('Waste management practices around the world', 'Waste Management'),
    ('Advancements in quantum computing technology', 'Technology'),
    ('Innovative materials for sustainability', 'Sustainability'),
    ('AI for predictive analytics', 'AI'),
    ('The role of AI in smart cities', 'AI'),
    ('Robots in space exploration', 'Robotics'),
    ('Landfill management practices', 'Waste Management'),
    ('Future trends in mobile technology', 'Technology'),
    ('Sustainable energy solutions', 'Sustainability'),
    ('AI-driven market research', 'AI'),
    ('The future of robotic surgery', 'Robotics'),
    ('Public awareness on waste management', 'Waste Management'),
    ('Tech innovations in smart homes', 'Technology'),
    ('The importance of carbon footprint reduction', 'Sustainability'),
]

df_sample = pd.DataFrame(sample_data, columns=['text', 'topic'])

df_sample['text'] = df_sample['text'].apply(preprocess_text)

label_encoder = LabelEncoder()
df_sample['encoded_topic'] = label_encoder.fit_transform(df_sample['topic'])

X_train, X_test, y_train, y_test = train_test_split(df_sample['text'], df_sample['encoded_topic'], test_size=0.2, random_state=42)

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english'),
    xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
)

model.fit(X_train, y_train)

predicted_labels = model.predict(X_test)
accuracy = accuracy_score(y_test, predicted_labels)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, predicted_labels))

param_grid = {
    'xgbclassifier__max_depth': [3, 5, 7, 10],
    'xgbclassifier__n_estimators': [50, 100, 200],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(model, param_grid, cv=4, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

predicted_labels_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, predicted_labels_best)

print(f"Best Model Accuracy: {accuracy_best * 100:.2f}%")
print("\nBest Model Classification Report:\n", classification_report(y_test, predicted_labels_best))

if 'clean_content' in articles_df.columns:
    articles_df['clean_content'] = articles_df['clean_content'].apply(preprocess_text)

    articles_df['predicted_topic'] = best_model.predict(articles_df['clean_content'])

    articles_df.to_csv('classified_articles.csv', index=False)
    print("Classified articles saved to classified_articles.csv")
else:
    print("Error: 'clean_content' column not found in articles_df.")
