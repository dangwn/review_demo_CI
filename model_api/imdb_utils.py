import nltk
import mlflow
import os
from nltk import word_tokenize

nltk.download("punkt")

def tokenize(text : str):
    '''
    Takes in a text string and returns a list where each item corresponds to a token.
    '''
    return word_tokenize(text)

def get_vector_headers(filepath : str):
    with open(filepath, 'r') as f:
        raw_words = f.readlines()
    words = []
    for w in raw_words:
        words.append(w[:-1])
    return words 

def vectorize_doc(doc:str, file_path:str):
    vector_headers = get_vector_headers(file_path)

    review = tokenize(doc)

    review_vector = []
    for ind, w in enumerate(vector_headers):
        review_vector.append(0)
        for r in review:
            if r.lower() == w.lower():
                review_vector[ind] += 1
    
    return review_vector





