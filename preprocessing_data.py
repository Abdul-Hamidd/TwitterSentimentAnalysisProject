# Importing essential libraries for data analysis and visualization

# 'pandas' is used for data manipulation and analysis, specifically to handle structured data in DataFrames.
import pandas as pd

# 'matplotlib.pyplot' is a plotting library used to create static, animated, and interactive visualizations in Python.
import matplotlib.pyplot as plt

# 'seaborn' is built on top of Matplotlib and provides a high-level interface for drawing attractive statistical graphics.
import seaborn as sns

"""
External function import:
Here, 'preprocess_texts' is being imported from a custom module 'utils'. 
This function is used to preprocess the text data (like cleaning, tokenization, etc.).

"""
from utils import preprocess_texts

# File path of the dataset to be loaded
data_file_path = r"C:\Users\Al Ghanii Computer\Desktop\Project_Datas\Un_Processed  data\Twitter_data_Oct2024.csv"

# Loading CSV data into a DataFrame using pandas
# Note: The dataset is loaded with ISO-8859-1 encoding to handle non-UTF characters.
data = pd.read_csv(data_file_path, header=None, encoding="ISO-8859-1")
print(f"Data loaded with shape: {data.shape}")  # Logging the shape of the dataset

# Display the first few rows of the raw data for inspection
print("First few rows of the raw data:")
print(data.head())  # Showing the initial rows of the dataset

# Dropping unnecessary columns from the dataset
# Assuming we need only two columns: sentiment and text. Adjust indices if necessary.
data = data.drop(data.columns[1:5], axis=1)  # Dropping columns 1 to 4
print(f"Data shape after dropping unnecessary columns: {data.shape}")  # Logging new shape after dropping columns

# Shuffling the data for randomization
# Setting random_state to ensure reproducibility
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
print("Data shuffled for randomization.")  # Confirming data shuffle

# Plotting the distribution of sentiment labels
# This plot visualizes how the sentiments (assumed to be in the first column) are distributed in the dataset.
plt.figure(figsize=(8, 6))  # Setting the plot size
sns.countplot(x=data[0], data=data)  # Creating a count plot for sentiment labels
plt.title("Distribution of Sentiment in Raw Data")  # Setting the plot title
plt.xlabel("Sentiment (0 = negative, 4 = positive)")  # X-axis label
plt.ylabel("Count")  # Y-axis label
plt.show()  # Displaying the plot

# Preprocessing the text column using the imported 'preprocess_texts' function
"""
Preprocess the text data:
- The text data is cleaned, tokenized, lemmatized, and stop words are removed.
- The 'preprocess_texts' function takes care of all the preprocessing steps.
"""
data[data.columns[1]] = preprocess_texts(data[data.columns[1]].to_numpy())
print("Text preprocessing completed.")  # Logging after preprocessing

# Save the preprocessed data into a new CSV file
"""
Save processed data:
- The cleaned text data is saved to a new CSV file.
- Encoding is set to 'utf-8' to ensure proper character encoding for the saved file.
"""
processed_file_path = r"C:\Users\Al Ghanii Computer\Desktop\Project_Datas\Processed Data\Twitter_data_Oct2024.csv"
data.to_csv(processed_file_path, index=False, header=False, encoding="utf-8")
print(f"Preprocessed data saved to: {processed_file_path}")  # Logging the file save operation

# Checking the first few rows of the cleaned data
print("First 5 lines of cleaned data:")
print(data.head())  # Displaying the first 5 rows of the cleaned data
