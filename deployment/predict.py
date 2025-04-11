'''
==========================================================================================================================================

Book Genre Prediction

Name: Celine Clarissa

Original Dataset: https://www.kaggle.com/datasets/athu1105/book-genre-prediction/data

Deployment: https://huggingface.co/spaces/celineclarissa/GC7

GitHub: https://github.com/celineclarissa/Book-Genre-Prediction


Background

As a data scientist at a book distribution company, understanding the characteristics of books is essential for accurately classifying
them by genre. This classification enables the company to develop targeted strategies and make informed decisions based on genre-specific
trends and preferences.

Problem Statement and Objectives

As a data scientist at a book distribution company, it is essential to develop skills in training, testing, tuning, and evaluating machine learning models. These models can help predict a book’s genre based on its summary before the company decides whether to distribute it. This enables the company to align its distribution strategy with genre-specific market demands.

The process begins with exploratory data analysis (EDA) to uncover patterns in book genres, followed by feature engineering to enhance the
dataset. A predictive model is then built using an Artificial Neural Network (ANN), with ongoing efforts to optimize its performance. The
goal is to achieve an accuracy above 90%, after which the final model will be deployed on Hugging Face within seven working days. The web
application will also include a dedicated page for interactive EDA visualizations.

==========================================================================================================================================
'''

# import libraries
import streamlit as st
import pandas as pd
import tensorflow as tf
import tensorflow_hub as tf_hub
from tensorflow.keras.models import load_model
import re
import numpy as np

# import feature engineering
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_hub as tf_hub
import warnings
warnings.filterwarnings('ignore')

# load preprocessor
# define stopwords
stopwords_eng = stopwords.words('english')
# define text preprocessing function
def text_preprocessing(text):
  '''
  This function is created to do text preprocessing: change text to lowercase, remove numbers and punctuation symbols, remove stopwords,
  lemmatize text, and tokenize text. Text preprocessing can be done just by calling this function.
  '''
  # change text to lowercase
  text = text.lower()
  # remove [UNK]
  text = text.replace('[UNK]', '')
  text = text.replace('unk', '')
  text = text.replace('UNK', '')
  text = text.replace('[unk]', '')
  # remove numbers
  text = re.sub(r'\d+', '', text)
  # remove comma
  text = text.replace(',', '')
  # remove period symbol
  text = text.replace('.', '')
  # remove exclamation mark
  text = text.replace('!', '')
  # remove question mark
  text = text.replace('?', '')
  # change texts using quotation marks that have negative connotation
  text = text.replace("don't", "do not")
  text = text.replace("aren't", "are not")
  text = text.replace("isn't", "is not")
  text = text.replace("didn't", "did not")
  text = text.replace("can't", "cannot")
  text = text.replace("couldn't", "could not")
  text = text.replace("didn't", "did not")
  # remove quotation mark
  text = text.replace('"', '')
  text = text.replace("'", '')
  text = text.replace('’', '')
  # remove whitespace
  text = text.strip()
  # tokenization
  tokens = word_tokenize(text)
  # remove stopwords
  tokens = [word for word in tokens if word not in stopwords_eng]
  # lemmatization
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  # combine tokens
  text = ' '.join(tokens)
  return text

# get pretrained layer from kaggle
url = 'https://tfhub.dev/google/tf2-preview/nnlm-id-dim128-with-normalization/1'
pretrained_layer = tf_hub.KerasLayer(url, output_shape=[128], input_shape=[], dtype=tf.string)

# load model
model = load_model('model_2.h5', custom_objects={'KerasLayer': pretrained_layer})

# define class dictionary
dict_class = {0: 'fantasy',
              1: 'science',
              2: 'crime',
              3: 'history',
              4: 'horror',
              5: 'thriller',
              6: 'psychology',
              7: 'romance',
              8: 'sports',
              9: 'travel'}

def run():
    # make title
    st.title('Book Genre Prediction')
    # insert image
    st.image('https://i.pinimg.com/originals/4a/a8/34/4aa834801140d2ce278c52dda94f2fc6.jpg', caption='Books (Source: Miranda on Pinterest)')

    # make form
    with st.form("G7_form"):

        st.write('### Insert data')

        # define each feature
        index = st.number_input('Index', min_value=0, max_value= 10000, value=4657)
        title = st.text_input(label='Input book title here.', value='The Notebook')
        summary = st.text_input(label='Input book title here.', value="Noah and Allie spend a wonderful summer together, but her family and the socio-economic realities of the time prevent them from being together. Although Noah attempts to keep in contact with Allie after they are forced to separate, his letters go unanswered. Eventually, Noah professes his undying and eternal love in one final letter. Noah travels north to find gainful employment and to escape the ghost of Allie, and eventually he goes off to war. After serving his country, he returns home to restore an old farmhouse. A newspaper article about his endeavor catches Allie's eye, and 14 years after she last saw Noah, Allie returns to him. The only problem is she is engaged to another man. After spending two wonderful reunion days together, Allie must decide between the two men that she loves.")

        # make submit button
        submitted = st.form_submit_button("Submit")

    # define inference data based on inputted data
    inf_data = {
    'index': index,
    'title': title,
    'summary': summary
}

    # make dataframe for inference data
    inf_data = pd.DataFrame([inf_data])

    # show inference data
    st.dataframe(inf_data)

    # create condition
    if submitted:

        ## preprocess text using function
        inf_data['text_processed'] = inf_data['summary'].apply(lambda x: text_preprocessing(x))

        ## define result using model
        result = model.predict(inf_data.text_preprocessed)

        ## take class with biggest probability
        result_class = np.argmax(result, axis=-1)

        ## print result
        st.write(f'#Book Genre Prediction: {dict_class[int(result_class)]}')

        ## show balloons after submitting
        st.spinner(text='Please wait for result')
        st.balloons()

# execute file
if __name__ == '__main__':
    run()