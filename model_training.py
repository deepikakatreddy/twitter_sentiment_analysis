import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

def train_model(input_file, output_model):
    df = pd.read_csv(input_file)
    
    X = df['clean_tweet']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    X_test = vectorizer.transform(X_test)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    
    with open(output_model, 'wb') as file:
        pickle.dump((model, vectorizer), file)
