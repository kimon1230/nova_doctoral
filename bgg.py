import json
import numpy as np
import re
from keras_core.preprocessing.sequence import pad_sequences

def preprocess_text(text):
    """
    Preprocesses raw text by converting to lowercase and removing non-alphabetic characters.
    
    Args:
        text (str): Raw text input
        
    Returns:
        list: List of preprocessed words
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Split into words
    words = text.split()
    return words

def load_data(path='bgg_data.json', num_words=None, skip_top=0, maxlen=None, 
              seed=113, start_char=1, oov_char=2, index_from=3, text_choice="review_text_preprocessed_noemoji"):
    """
    Loads and preprocesses review data from JSON file for sentiment analysis.
    
    Args:
        path (str): Path to JSON data file
        num_words (int, optional): Maximum number of words to keep
        skip_top (int): Number of most frequent words to skip
        maxlen (int, optional): Maximum length of each sequence
        seed (int): Random seed for reproducibility
        start_char (int): Character to mark sequence start
        oov_char (int): Character for out-of-vocabulary words
        index_from (int): Index offset for word indices
        text_choice (str): Which version of review text to use (raw, preprocessed, or no emoji)
        
    Returns:
        tuple: (X, y) where X is padded sequence data and y is sentiment labels (0=negative, 1=neutral, 2=positive)
    """
    # Load data from JSON
    with open(path, 'r') as f:
        data = json.load(f)

    reviews = []
    sentiments = []
    for dealership in data:
        for review in dealership['reviews']:
            reviews.append(review[text_choice])
            sentiments.append(review['review_sentiment'])

    # Convert sentiments to numerical labels
    sentiment_to_label = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = np.array([sentiment_to_label[s] for s in sentiments])

    # Get word index
    word_index = get_word_index(path, text_choice)

    # Convert words to indices
    X = [[word_index.get(word, oov_char) for word in review.split()] for review in reviews]

    # Apply word index filters based on `num_words` and `index_from`
    if num_words is not None:
        X = [[start_char] + [w + index_from if (skip_top <= w < num_words) else oov_char for w in seq] for seq in X]

    # Pad sequences
    X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post')

    return X, y

def get_word_index(path='bgg_data.json', text_choice="review_text_preprocessed_noemoji"):
    """
    Creates a word-to-index mapping from the review data.
    
    Args:
        path (str): Path to JSON data file
        text_choice (str): Which version of review text to use
        
    Returns:
        dict: Dictionary mapping words to their indices, starting from index 1
    """
    with open(path, 'r') as f:
        data = json.load(f)

    word_index = {}
    index = 1  # Start from 1 as 0 is typically reserved for padding
    for dealership in data:
        for review in dealership['reviews']:
            for word in review[text_choice].split():
                if word not in word_index:
                    word_index[word] = index
                    index += 1

    return word_index