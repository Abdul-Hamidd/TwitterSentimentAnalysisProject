"""
Neural Network for Binary Classification on Twitter Sentiment Data
This script trains a neural network to classify Twitter sentiment (positive/negative) using TF-IDF features.
Author: Abdul Hamid
Date: October 2024
"""

# Standard libraries for system encoding and handling
import sys  # Provides access to system-specific parameters and functions
import os   # Allows interaction with the operating system, including file and directory manipulation
import io   # Supports the handling of various types of I/O, including reading and writing to files


# Numerical and data handling libraries
import numpy as np
import pandas as pd

# Import TfidfVectorizer from sklearn
# TfidfVectorizer is used to convert a collection of raw documents into a matrix of TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer

# Import pickle to save and load the TfidfVectorizer
# This helps in reusing the same vectorizer during both training and prediction
import pickle


# Splitting data into train, validation, and test sets
from sklearn.model_selection import train_test_split

# Performance metrics for model evaluation
from sklearn.metrics import accuracy_score
 
# Suppress TensorFlow logging messages to only show errors (level 3)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# TensorFlow for building and training neural networks
import tensorflow as tf

# Optimizer, layers, and model utilities from Keras API in TensorFlow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal

# Callbacks for early stopping, model checkpointing, and learning rate adjustment
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set output encoding to UTF-8 for better compatibility with different systems
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Step 1: Load and preprocess the data 
Loading the processed data file and transforming the labels.
"""
# Load processed data from CSV
data = pd.read_csv(r"C:\Users\Al Ghanii Computer\Desktop\Project_Datas\Processed Data\Twitter_data_Oct2024.csv", 
                   header=None, encoding="utf-8")


"""
 Replace labels: Convert label 4 to 1 for binary classification
 This means that we treat label 4 (positive sentiment) as 1,
 while retaining label 0 (negative sentiment) as it is.
"""
 # Replace labels: convert label 4 to 1 for binary classification (1: positive, 0: negative)
data[data.columns[0]] = data[data.columns[0]].replace(4, 1)

"""
# Apply tfidf to the text column for feature extraction
# The second column (data.columns[1]) contains the text data that we want to vectorize.
# The tfidf function returns the TF-IDF matrix (x) and the vectorizer object.
"""

# Define and initialize the TfidfVectorizer
# This vectorizer will transform the text data into numerical form using Term Frequency and Inverse Document Frequency (TF-IDF)
vectorizer = TfidfVectorizer(max_features=30000)

# Fit the TfidfVectorizer to the training data (X_train) and convert it into TF-IDF features
# X_train is the training dataset that contains text data
x = vectorizer.fit_transform(data[data.columns[1]])

# Save the fitted TfidfVectorizer to a file so that it can be used later during prediction
# This ensures the same vocabulary is used for transforming new data
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)


# Extract labels (target values)
y = data[data.columns[0]].values

"""
Step 2: Split Data into Training, Validation, and Test Sets
Splitting the data into training (60%), validation (20%), and testing (20%).
"""
# Split data into training + validation and testing sets(20% test data , 20% val_data , 60% train data)
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Further split the training + validation set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42)

# Only train the model if this script is run directly
if __name__ == "__main__":

    """
    Step 3: Model Creation
    Defining a neural network model with HeNormal initialization, dropout layers for regularization, and sigmoid activation for binary classification.
    """
    # Build the neural network model
    model = Sequential([
        Input(shape=(x_train.shape[1],)),
        
        # First dense layer with L2 regularization and ReLU activation
        Dense(128, activation='relu', kernel_initializer=HeNormal()),
        Dropout(0.2),
        
        # Second dense layer
        Dense(64, activation='relu', kernel_initializer=HeNormal()),
        Dropout(0.1),
        
        # Third dense layer
        Dense(32, activation='relu', kernel_initializer=HeNormal()),
        Dropout(0.2),
        
        # Output layer with sigmoid activation for binary classification
        Dense(1, activation='sigmoid')
    ])

    """
    Step 4: Compile the Model
    Compiling the model with Adam optimizer and binary crossentropy loss function for binary classification.
    """
    # Compile the model with Adam optimizer and binary crossentropy loss function
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"]
    )

    """
    Step 5: Define Callbacks for Efficient Training
    Early stopping, learning rate reduction, and model checkpointing to prevent overfitting and reduce learning rate when needed.
    """
    # Early stopping to stop training if validation loss doesn't improve for 3 epochs
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Reduce learning rate by a factor of 0.2 if validation loss does not improve for 2 epochs
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.00001
    )

    # Save the best model based on validation loss
    model_checkpoint = ModelCheckpoint(
        'Trained_model.keras', # save model in 'saved_model' directory
        monitor='val_loss',
        save_best_only=True
    )

    """
    Step 6: Train the Model
    Training the model with training and validation sets and the defined callbacks.
    """
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=256,
        validation_data=(x_val, y_val),  # Add validation data for evaluation
        callbacks=[early_stopping, reduce_lr, model_checkpoint],  # Use defined callbacks
        verbose=1  # Set verbose=1 to see progress during training
    )

    """
    Step 7: Evaluate Model Performance
    Print the final training and validation accuracy and loss after training.
    """
    # Print final training and validation accuracy/loss
    print(f"Final training accuracy: {history.history['accuracy'][-1] * 100:.2f}%")        #Expected Training_accuracy = 82.64%
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")                        #Expected Training_loss = 0.36
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")    #Expected Val_accuracy = 78%
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")                    #Expected val_loss = 0.46
