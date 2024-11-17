import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
data = pd.read_csv(r"C:\Users\91781\Desktop\Batch-2-A\Fake_Profile_Identification_code\fake_profile_identification\Profile_Datasets.csv")

# Preprocess data if needed
data['description'].fillna('', inplace=True)
data['name'].fillna('', inplace=True)
data['followers_count'].fillna('', inplace=True)
data['friends_count'].fillna('', inplace=True)
data['location'].fillna('', inplace=True)

# Combine features
data['combined_features'] = data['name'] + ' ' + data['followers_count'].astype(str) + ' ' + data['friends_count'].astype(str) + ' ' + data['location'] + ' ' + data['description']

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and punctuations, and apply lemmatization
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Preprocess the combined features
data['preprocessed_features'] = data['combined_features'].apply(preprocess_text)

# Split dataset into features and labels
X = data['preprocessed_features']
y = data['Label']

# Text vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train classifiers
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

svm_predictions = svm_classifier.predict(X_test)
nb_predictions = nb_classifier.predict(X_test)
rf_predictions = rf_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)


# Calculate evaluation metrics
svm_accuracy = accuracy_score(y_test, svm_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

svm_precision = precision_score(y_test, svm_predictions)
nb_precision = precision_score(y_test, nb_predictions)
rf_precision = precision_score(y_test, rf_predictions)
knn_precision = precision_score(y_test, knn_predictions)

svm_recall = recall_score(y_test, svm_predictions)
nb_recall = recall_score(y_test, nb_predictions)
rf_recall = recall_score(y_test, rf_predictions)
knn_recall = recall_score(y_test, knn_predictions)

svm_f1 = f1_score(y_test, svm_predictions)
nb_f1 = f1_score(y_test, nb_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
knn_f1 = f1_score(y_test, knn_predictions)

# Print or display evaluation metrics
print("SVM Metrics:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)
print(classification_report(y_test, svm_predictions))

print("Naive Bayes Metrics:")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
print(classification_report(y_test, nb_predictions))

print("Random Forest Metrics:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print(classification_report(y_test, rf_predictions))

print("KNN Metrics:")
print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)
print(classification_report(y_test, knn_predictions))

# Save vectorizer and classifiers
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(svm_classifier, 'svm_classifier.pkl')
joblib.dump(nb_classifier, 'nb_classifier.pkl')
joblib.dump(rf_classifier, 'rf_classifier.pkl')
joblib.dump(knn_classifier, 'knn_classifier.pkl')

# Define the Streamlit app
def main():
    st.title('Fake Profile Identification')

    # Text input for the user to enter profile description
    name = st.text_input('Enter name:')
    followers_count = st.text_input('Enter followers count:')
    friends_count = st.text_input('Enter friends count:')
    location = st.text_input('Enter location:')
    description = st.text_area('Enter profile description:')

    # Prediction button
    if st.button('Predict'):
        # Load vectorizer and classifier
        vectorizer = joblib.load('vectorizer.pkl')
        svm_classifier = joblib.load('svm_classifier.pkl')

        # Preprocess the input description
        combined_features = name + ' ' + followers_count + ' ' + friends_count + ' ' + location + ' ' + description
        description_vectorized = vectorizer.transform([preprocess_text(combined_features)])
        prediction = svm_classifier.predict(description_vectorized)
        
        # Display prediction
        if prediction[0] == 0:
            st.error('Prediction: Fake Profile')
        else:
            st.success('Prediction: Genuine Profile')


if __name__ == '__main__':
    main()