import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Read data from CSV
def read_csv(filename):
    df = pd.read_csv(filename, usecols=["v1", "v2"], encoding='latin1')  # Adjust the encoding if needed
    return df['v2'], df['v1']

# Train SVM model
def train_model(train_texts, train_labels):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
    train_features = tfidf_vectorizer.fit_transform(train_texts)
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(train_features, train_labels)
    return tfidf_vectorizer, svm_classifier

# Predict label for a single message
def predict_label(message, tfidf_vectorizer, svm_classifier):
    message_features = tfidf_vectorizer.transform([message])
    prediction = svm_classifier.predict(message_features)
    return prediction[0]

# Main function
def main():
    # Sample usage
    csv_path = input("Enter the path to the CSV file: ")
    train_texts, train_labels = read_csv(csv_path)
    tfidf_vectorizer, svm_classifier = train_model(train_texts, train_labels)
    print("Model trained successfully!")

    # Real-time spam detection
    while True:
        message = input("Enter a message to classify (or 'exit' to quit): ")
        if message.lower() == 'exit':
            print("Exiting...")
            break
        else:
            prediction = predict_label(message, tfidf_vectorizer, svm_classifier)
            if prediction == 'spam':
                print("SPAM Detected!")
            else:
                print("HAM Message")
            
# Run the main function
if __name__ == "__main__":
    main()
