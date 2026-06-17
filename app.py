import streamlit  as st
import pickle
import nltk
import sklearn

from nltk.stem.porter import PorterStemmer as ps
import string
from nltk.corpus import stopwords


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        steamed = ps.stem(i)
        if steamed not in y:
            y.append(steamed)

    return " ".join(y)

tfidf=pickle.load(open('vector.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title('Spam Classification')


input_sms=st.text_input('Enter  the message ')

if st.button('Classify'):

    # preprocessing
    transform_text=transform_text(input_sms)

    # vectorizztion
    vector_input=tfidf.transform([transform_text])


    # predict
    result=model.predict(vector_input)[0]

    if result==1:
        st.header('spam')
    else:
        st.header('not spam')