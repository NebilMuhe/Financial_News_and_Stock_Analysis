import pandas as pd


class DescriptiveAnalysis:
    def __init__(self, data):
        self.data = data
    

    def analyze_headline_length(self):
        self.data['headline_length'] = self.data['headline'].apply(len)
        length_analytic = self.data['headline_length'].describe()
        return length_analytic
    

    def count_number_of_articles_per_publisher(self):
        articles_per_publishers = self.data['publisher'].value_counts()
        return articles_per_publishers
    

    def analyze_publication_date(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        news_frequency = self.data['date'].dt.date.value_counts().sort_index()
        return news_frequency