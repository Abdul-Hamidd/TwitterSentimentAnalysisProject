"""
first of the all we initialize requrements 
these are necessary tools to
"""
import numpy as np

import re  # Regular expressions library for text cleaning
import nltk  # Natural Language Toolkit for text preprocessing
from nltk.corpus import stopwords, wordnet  # Stopwords and WordNet corpus from NLTK
from nltk.stem import WordNetLemmatizer  # Lemmatizer for reducing words to their base form
from nltk import pos_tag  # Part-of-speech tagging for words
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vectorizer for feature extraction

# Download necessary NLTK resources (these lines ensure the required NLTK data is available)
nltk.download('averaged_perceptron_tagger', quiet=True)  # POS tagger needed for lemmatization
nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet for lemmatizer
nltk.download('stopwords', quiet=True)  # Download English stopwords list
nltk.download('wordnet', quiet=True)  # Download WordNet corpus for lemmatization

# Preprocessing setup: initializing stopwords and the lemmatizer
stop_words = set(stopwords.words("english"))  # Load set of common English stopwords
lemmatizer = WordNetLemmatizer()  # Create an instance of the WordNetLemmatizer
#print("Preprocessing setup completed: Stop words and lemmatizer initialized.")  # Log successful setup

def get_wordnet_pos(word):
    """
    Convert NLTK POS tags to WordNet POS tags for more accurate lemmatization.
    - NLTK POS tags are mapped to WordNet's format (adjective, verb, noun, adverb).
    - This function improves the lemmatization by using the right POS tag.
    """
    # Get the first character of the POS tag (J: adjective, V: verb, etc.)
    tag = pos_tag([word])[0][1][0].upper()
    tag_dic = {
        "J": wordnet.ADJ,  # Adjective
        "V": wordnet.VERB,  # Verb
        "N": wordnet.NOUN,  # Noun
        "R": wordnet.ADV  # Adverb
    }
    # Return the mapped WordNet POS tag or default to noun
    return tag_dic.get(tag, wordnet.NOUN)

def preprocess_texts_fun(texts):
    """
    Preprocess the input texts:
    - Lowercases text
    - Removes punctuation and special characters using regex
    - Tokenizes the text into individual words
    - Lemmatizes the tokens based on their POS tags
    - Removes stopwords (common words that don't add much value)
    - Rejoins the lemmatized tokens back into a single string
    
    Arguments:
    - texts: list of text strings to preprocess
    
    Returns:
    - A list of preprocessed text strings
    """
    processed_texts = []  # List to store processed text
    
    for text in texts:
        # Skip any non-string entries by adding an empty string placeholder
        if not isinstance(text, str):
            processed_texts.append("")  # Handle non-string entries
            continue
        
        # Step 1: Lowercase the text and clean using regex (removes punctuation)
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[^\w\s]", " ", text)  # Replace special characters with a space
        
        # Step 2: Tokenize and remove stopwords
        tokens = text.split()  # Split text into words (tokens)
        
        # Step 3: Lemmatize tokens with the correct POS tag and remove stopwords
        lemmatized_tokens = [
            lemmatizer.lemmatize(word, get_wordnet_pos(word))  # Lemmatize word with its POS tag
            for word in tokens if word not in stop_words  # Exclude stopwords
        ]
        
        # Step 4: Rejoin the lemmatized tokens into a single string
        proc_text = " ".join(lemmatized_tokens)
        
        # Step 5: If lemmatized text is empty, fall back to the original cleaned text
        if not proc_text.strip():  # If the processed text is empty
            processed_texts.append(text)  # Add original cleaned text as fallback
        else:
            processed_texts.append(proc_text)  # Otherwise, add the processed text
    
    return processed_texts  # Return the list of processed texts

def preprocess_texts(texts):
    """
    Wrapper function for the `preprocess_texts_fun` that processes a list of texts.
    - Applies text preprocessing and logs the number of texts processed.
    
    Arguments:
    - texts: list of text strings to preprocess
    
    Returns:
    - List of processed text strings
    """

    processed_texts = preprocess_texts_fun(texts)  # Call the preprocessing function
    return processed_texts  # Return the processed texts

