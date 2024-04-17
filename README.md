Objective->
Build an AI model that can classify SMS messages as spam or legitimate. Use techniques like TF-IDF or word
embeddings with classifiers like Naive Bayes, Logistic Regression, or Support Vector Machines to identify spam
messages

Dataset:
The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It
contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

Proposed Solution 


This code implements a simple spam classifier using a Support Vector Machine (SVM) model trained on TF-IDF (Term Frequency-Inverse Document Frequency) features. Here's a step-by-step explanation of the code and its processes:

Imports:

import pandas as pd: Imports the Pandas library for data manipulation.
from sklearn.feature_extraction.text import TfidfVectorizer: Imports the TF-IDF vectorizer from scikit-learn, which will convert text data into numerical TF-IDF features.
from sklearn.svm import SVC: Imports the Support Vector Classification (SVC) model from scikit-learn.

Function Definitions:

read_csv(filename): Reads data from a CSV file. It takes the filename as input, reads the CSV, and returns the text data (v2) and corresponding labels (v1).
train_model(train_texts, train_labels): Trains the SVM model. It takes the text data and labels as input, converts text data into TF-IDF features, and fits an SVM classifier to these features.
predict_label(message, tfidf_vectorizer, svm_classifier): Predicts the label (spam or ham) for a single message. It takes the message, TF-IDF vectorizer, and SVM classifier as input, converts the message into TF-IDF features using the vectorizer, and predicts the label using the SVM classifier.
main(): The main function of the script. It orchestrates the entire process, from reading data to training the model, to real-time classification.

Main Function:

Asks the user to input the path to the CSV file containing the training data.
Reads the CSV file and extracts the text data and corresponding labels.
Trains the SVM model on the training data.
Enters a loop for real-time spam detection.
Asks the user to input a message to classify.
If the user inputs "exit", the program exits.
Otherwise, it predicts whether the message is spam or ham using the trained model and prints the result.
Running the Script:

The if __name__ == "__main__": block ensures that the main() function is executed when the script is run as the main program.

Output:
 IMPORTANT--------------------> you have to give the  path of the CSV file in the output panel 

let's break down the process of obtaining the output step by step:

Reading CSV File:

The program starts by asking the user to input the path to the CSV file containing the training data.
Once the user provides the path, the program reads the CSV file using the read_csv() function.
The function extracts the text data (v2) and corresponding labels (v1) from the CSV file.

Training the Model:

After reading the data, the program proceeds to train the SVM model using the train_model() function.
This function takes the extracted text data and labels as input.

Inside the function:
It initializes a TF-IDF vectorizer (tfidf_vectorizer) with a maximum of 5000 features.
The text data is converted into TF-IDF features using the fit_transform() method of the vectorizer.
An SVM classifier (svm_classifier) with a linear kernel is initialized.
The classifier is trained on the TF-IDF features and corresponding labels using the fit() method.

Real-time Spam Detection:
After the model is trained, the program enters a loop for real-time spam detection.

Inside the loop:
The user is prompted to enter a message to classify.
If the user inputs "exit", the loop is terminated, and the program exits.
Otherwise, the message is passed to the predict_label() function along with the TF-IDF vectorizer and SVM classifier.

Inside the predict_label() function:
The input message is converted into TF-IDF features using the transform() method of the vectorizer.
The SVM classifier predicts the label (spam or ham) for the message.
The predicted label is returned.
B
ased on the predicted label returned by the predict_label() function:
If the label is "spam", the program prints "SPAM Detected!".
If the label is "ham" (not spam), the program prints "HAM Message".
Exiting the Program:
If the user inputs "exit" during the real-time spam detection phase, the program prints "Exiting..." and exits the loop, thereby terminating the program.
In summary, the output is obtained by iteratively classifying user-input messages in real-time using the trained SVM model and providing the corresponding classification result ("SPAM Detected!" or "HAM Message").
![Screenshot 2024-04-17 231301](https://github.com/SATYAMSINGH0707/Aframe-Technologies/assets/97894680/ef33e2e4-8f93-48bf-93ad-ec6829007a91)

