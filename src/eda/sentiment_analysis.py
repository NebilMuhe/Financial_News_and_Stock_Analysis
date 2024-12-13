
from textblob import TextBlob

import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
import nltk

nltk.download('punkt')
nltk.download('stopwords')


class SentimentAnalysis:
    def __init__(self, data):
        self.data = data
        self.stop_words = set(stopwords.words('english'))

    def perform_sentiment_analysis(self):
        self.data['sentiment'] = self.data['text'].apply(self.get_sentiment)
        return self.data['sentiment'].value_counts()

    def get_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    
    def clean_data(self, data):
        # Lowercase the text
        data = data.lower()
        # Remove special characters, numbers, and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def preprocess(self, df):
        df[self.data] = df[self.data].apply(self.clean_data)
        return df
    
    
    def extract_keywords(self, df, ngram_range=(1, 2), top_n=20, method=''):
        vectorizer = TfidfVectorizer if method == 'tfidf' else CountVectorizer
        vec = vectorizer(ngram_range=ngram_range, max_features=5000)
        matrix = vec.fit_transform(df[self.data])

        # Sum up word frequencies or scores
        scores = matrix.sum(axis=0).A1
        keywords = [(word, scores[idx]) for word, idx in vec.vocabulary_.items()]
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

        return keywords[:top_n]
    

    def extract_topics(self, df, n_topics=5, n_words=10):
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        matrix = vectorizer.fit_transform(df[self.data])
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics.append(f"Topic {idx + 1}: {', '.join(top_words)}")
        return topics
