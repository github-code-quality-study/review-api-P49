import csv
import json
import os
from uuid import uuid4
from datetime import datetime
from typing import Callable, Any
from wsgiref.simple_server import make_server

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs

# Initialize NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Load data
reviews = pd.read_csv('data/reviews.csv')
reviews_dict = reviews.to_dict('records')

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Define allowed locations
ALLOWED_LOCATIONS = {
    'Albuquerque, New Mexico',
    'Carlsbad, California',
    'Chula Vista, California',
    'Colorado Springs, Colorado',
    'Denver, Colorado',
    'El Cajon, California',
    'El Paso, Texas',
    'Escondido, California',
    'Fresno, California',
    'La Mesa, California',
    'Las Vegas, Nevada',
    'Los Angeles, California',
    'Oceanside, California',
    'Phoenix, Arizona',
    'Sacramento, California',
    'Salt Lake City, Utah',
    'San Diego, California',
    'Tucson, Arizona'
}

def analyze_sentiment(review_body: str) -> dict:
    sentiment_scores = sia.polarity_scores(review_body)
    return sentiment_scores

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.reviews = reviews_dict

    def filter_reviews(self, location=None, start_date=None, end_date=None):
        filtered_reviews = self.reviews

        # Filter by location
        if location:
            if location not in ALLOWED_LOCATIONS:
                return []
            filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

        # Filter by date range
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date]
        
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]
        
        # Add sentiment analysis
        for review in filtered_reviews:
            review['sentiment'] = analyze_sentiment(review['ReviewBody'])
        
        # Sort by compound sentiment score
        sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)
        
        return sorted_reviews

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        path = environ.get('PATH_INFO', '')
        query_string = environ.get('QUERY_STRING', '')
        request_method = environ.get('REQUEST_METHOD', '')

        # Handle GET request
        if request_method == "GET":
            params = parse_qs(query_string)
            location = params.get('location', [None])[0]
            start_date = params.get('start_date', [None])[0]
            end_date = params.get('end_date', [None])[0]

            reviews = self.filter_reviews(location, start_date, end_date)
            response_body = json.dumps(reviews, indent=2).encode('utf-8')

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        # Handle POST request
        elif request_method == "POST":
            content_length = int(environ.get('CONTENT_LENGTH', 0))
            request_body = environ['wsgi.input'].read(content_length).decode('utf-8')
            params = parse_qs(request_body)

            location = params.get('Location', [None])[0]
            review_body = params.get('ReviewBody', [None])[0]

            # Validate input
            if not location or not review_body or location not in ALLOWED_LOCATIONS:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [b'{"error": "Invalid input"}']

            # Create a new review
            new_review = {
                'ReviewId': str(uuid4()),
                'Location': location,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ReviewBody': review_body,
                'sentiment': analyze_sentiment(review_body)
            }

            # Append new review to in-memory list
            self.reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode('utf-8')
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        # Handle unknown methods
        else:
            start_response("405 Method Not Allowed", [("Content-Type", "application/json")])
            return [b'{"error": "Method not allowed"}']

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
