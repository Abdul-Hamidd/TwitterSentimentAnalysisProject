# Twitter Sentiment Analysis Project

This project aims to analyze the sentiment of tweets using machine learning techniques. The primary goal is to classify tweets as positive, or negative, based on their content.

### About the Dataset

This project utilizes the **Sentiment140** dataset, which contains **1,600,000 tweets** extracted using the Twitter API(***Download via Kaggle***) . The tweets have been annotated for sentiment analysis, with the following labels:

- **0** = Negative
- **4** = Positive

### Dataset Fields

The dataset contains the following six fields:

- **target**: The polarity of the tweet (0 = negativ, 4 = positive)
- **ids**: The unique identifier of the tweet (e.g., 2087)
- **date**: The timestamp of when the tweet was posted (e.g., Sat May 16 23:58:44 UTC 2009)
- **flag**: The query used to collect the tweet (if no query was used, this value is `NO_QUERY`)
- **user**: The username of the individual who tweeted (e.g., robotickilldozr)
- **text**: The content of the tweet (e.g., Lyx is cool)

### Total Size**: **238.8 MB**
## Format**: **CSV**




### Project Overview

In today's digital age, analyzing sentiments expressed in social media is crucial for brands, businesses, and researchers. This project provides a framework to automate the sentiment analysis of tweets, making it easier to gauge reactions and sentiments at scale.

### Project Structure

The project consists of four main components:

***utils.py***:

Contains utility functions for preprocessing tweets, including cleaning text, tokenization, and removing stop words.

***Preprocessing_data.py****:

 contains preprocessing tweets, including cleaning text, tokenization, and removing stop words by using preprocessing function defined in utils.py .

***Train_Model.py***:

Responsible for training the sentiment analysis model using a dataset of tweets. It handles the transformation of text data into TF-IDF features and saves both the trained model and vectorizer for later use.
(***TF-IDF***:tfidf_vectorizer.pkl ***with pkl format***, ***Trained model***:Trained_model.keras ***with keras format***

***prediction.py***:

Provides an interactive command-line interface to predict the sentiment of new tweets in real-time. Users can input tweets and receive immediate feedback on their sentiment.


### Installation and Requirements

To run this project, ensure you have Python 3.x installed along with the required libraries. You can install the necessary packages using the following command:

***pip install numpy tensorflow scikit-learn***



### Predict Sentiment

Run the prediction.py script to start predicting sentiments. The program will prompt you to enter a tweet, and it will output the predicted sentiment.

***Example Usage***

Enter your tweet: "I love this app!"
The predicted sentiment for the tweet is: Encouraging Response


### Key Features

***Consistent TF-IDF Processing***: Ensures that the same TF-IDF vectorizer is used during both training and prediction phases, maintaining consistency in feature extraction.

***Real-Time Interaction***: Users can get immediate feedback on tweet sentiments, making it easy to test various inputs dynamically.


### Troubleshooting
If you encounter any issues, consider the following tips:

***Model Not Found***: Ensure that the files Trained_model.keras and tfidf_vectorizer.pkl are in the correct directory from which you are running prediction.py.
***Library Issues***: Make sure all required libraries are correctly installed and updated.
























