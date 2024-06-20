import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_tweet(tweet):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenization
    words = word_tokenize(tweet)
    
    # Remove stopwords and lemmatization
    filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    
    return ' '.join(filtered_words)

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df['clean_tweet'] = df['tweet'].apply(preprocess_tweet)
    df.to_csv(output_file, index=False)
