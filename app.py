import streamlit as st
import pandas as pd
import joblib

model_path = './models/LinSVC.joblib'
model = joblib.load(model_path)


def custom_preprocessor(test, vec):
    # same as what is done for training dataset
    test = test.str.lower()
    test = test.str.replace(r'[^\w\s]', '', regex=True)
    test = test.str.replace(r'\s+', ' ', regex=True)
    test = test.str.strip()
    test = vec.transform(test)
    return test


def prediction(model, test):
    if model.predict(test) == 1:
        return "spam"
    else:
        return "ham"


vectorizer = joblib.load('./models/vectorizer.joblib')
model = joblib.load('./models/LinSVC.joblib')

# Streamlit app
st.title("SMS Spam Classifier")

user_input = st.text_area("Enter the message text:")

if st.button("Classify"):
    if user_input:
        test = pd.Series([user_input])

        test = custom_preprocessor(test, vectorizer)

        result = prediction(model, test)

        st.write(f"The message is classified as: **{result}**")
    else:
        st.warning("Please enter a message text to classify.")
