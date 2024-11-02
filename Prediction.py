# Importing necessary libraries and modules for prediction

# Importing NumPy for numerical operations and working with arrays
import numpy as np

# Importing the OS module to interact with the operating system (e.g., file paths, directories)
import os


# Import pickle to load the saved TfidfVectorizer
# This ensures that the same vectorizer with the same vocabulary is used for prediction as was used during training
import pickle


# Importing the test dataset (x_test: input data, y_test: labels) from Train_Model
# y_test is imported to evaluate the model's performance if labels are available
from Train_Model import x_test, y_test  # Import y_test as well if you have labels

# Importing f1_score from sklearn to evaluate model performance
# F1 score is the harmonic mean of precision and recall, useful for imbalanced datasets
from sklearn.metrics import f1_score

"""
External function import:
Here, 'preprocess_texts' is being imported from a custom module 'utils'. 
This function is used to preprocess the text data (like cleaning, tokenization, etc.).
"""
from utils import preprocess_texts


# Load the saved TfidfVectorizer from the file to maintain consistency in text transformations during prediction
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Suppress TensorFlow logging messages to only show errors (level 3)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("Trained_model.keras")

# Display the shape of x_test and y_test data to verify correct import
##print("x_test data shape:", x_test.shape)
##print("y_test data shape:", y_test.shape)

#Model evaluation on x_test and y_test data
##loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

##print("Model evaluation---")
##print(f"x_test Loss: {loss:.2f}")                # Expected test_loss = 0.46
##print(f"x_test Accuracy: {accuracy*100:.2f}%")   # Expected Test_accuracy = 78%


# Model evaluation with F1 Score (using original test data)
##predictions = model.predict(x_test, verbose=0)
##binary_predictions = (predictions > 0.5).astype(int)

# Calculate the F1 Score to analyze the model's performance on unseen data
##f1 = f1_score(y_test, binary_predictions, average='binary')

##print(f"F1 Score on test data: {f1*100:.2f}%")    # Expected F1_score = 78%


# Function to preprocess, transform using TF-IDF,convert input to numpy array and predict sentiment for a new tweet
def predict_sentiment(new_tweet):
    """
    This function takes a new tweet as input, preprocesses it,
    converts it to TF-IDF features, and returns the model's predicted sentiment 
    (Encouraging Response or Disapproving Response).
    
    Parameters:
    new_tweet (str or list): A single tweet or a list of tweets to be analyzed.
    
    Returns:
    str: Predicted sentiment for the given tweet.
    """

    # Check if the input tweet is a string, if yes, convert it to a numpy array
    if isinstance(new_tweet, str):
        new_tweet = np.array([new_tweet])  # Convert single string to array for consistency

    # Preprocess the tweet to clean it and prepare it for prediction
    preprocessed_tweet = preprocess_texts(new_tweet)
    print(f"prepro: {preprocessed_tweet}")
    #print(f"Preprocessed tweet: {preprocessed_tweet}")  # Debugging: print the preprocessed tweet

    # Check if preprocessing returned an empty list (no valid tokens)
    if not preprocessed_tweet:
        print("No valid tokens found for prediction.")  # Debugging: No words found after preprocessing
        return "No valid sentiment prediction."  # Return an error message if nothing to predict

    # Transform the preprocessed tweet into TF-IDF features using the preloaded vectorizer
    tweet_tfidf = vectorizer.transform(preprocessed_tweet)
    print(f"tfidf: {tweet_tfidf}")

    # Check if the TF-IDF is empty (all zeroes)
    if tweet_tfidf.nnz == 0:  # nnz counts non-zero entries
        print("The TF-IDF features array is empty. Cannot make a prediction.")
        return "No valid sentiment prediction."

    

    # Print the shape of the TF-IDF array for debugging purposes
    #print(f"TF-IDF shape: {tweet_tfidf.shape}")  # Debugging: Display TF-IDF array dimensions

    # Try to make a prediction using the preloaded model
    try:
        prediction = model.predict(tweet_tfidf, verbose=0)  # Predict using the ML model

        # Convert the model's output (which is continuous) into a binary result using a threshold of 0.5
        binary_prediction = (prediction > 0.5).astype(int)  # Convert to 1 (positive) or 0 (negative)

        # Based on the binary prediction, assign the sentiment label
        sentiment = "Encouraging Response" if binary_prediction == 1 else "Disapproving Response"
        return sentiment  # Return the predicted sentiment

    except Exception as e:
        # Handle any errors during prediction (e.g., model issues or incorrect input format)
        print(f"Error during prediction: {e}")  # Debugging: Display the error message
        return "Prediction error."  # Return a generic error message if prediction fails




# Example usage:
# Prompt user for input and run in a loop for continuous predictions
while True:
    # Get user input for the tweet
    tweet = input("Enter your tweet (or type 'exit' to quit): ")

    # Check if user wants to exit the loop
    if tweet.lower() == 'exit':
        print("Exiting the sentiment prediction program. Goodbye!")  # Farewell message
        break  # Exit the loop if user types 'exit'

    # Predict the sentiment for the new tweet
    predicted_sentiment = predict_sentiment(tweet)

    # Output the result
    print(f"The predicted sentiment for the tweet is: {predicted_sentiment}")  # Display the prediction result
