'''
==========================================================================================================================================

Book Genre Prediction

Name: Celine Clarissa

Original Dataset: https://www.kaggle.com/datasets/athu1105/book-genre-prediction/data

Deployment: https://huggingface.co/spaces/celineclarissa/GC7

GitHub: https://github.com/celineclarissa/Book-Genre-Prediction


Background

I am a data scientist at a book distribution company. As a company, it is important to know the characteristics of books in order to sort
books based on its genre. The information can then be used to make strategies based on book genre.

Problem Statement and Objective

As a data scientist at a book distribution company, skills of training, testing, tuning, and evaluating a model are important because the
company can then use the model to predict the genre of a book before accepting to distribute it. The company can then determine a business
strategy like planning a choosing books to distribute based on genre, for example. This can be done by using data. After analyzing book
genre characteristics from EDA, data scientist will then do feature engineering towards data. Then, data scientist will do modelling with
ANN to predict genre of book. Then, data scientist will attempt to improve model. The best model is aimed to have an accuracy score of
more than 90% and then deployed on HuggingFace for effective use after 7 working days. Webapp where model is deployed will also feature
a page for EDA.

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