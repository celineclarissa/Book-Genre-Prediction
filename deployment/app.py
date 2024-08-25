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
import eda
import predict

# create sidebar to navigate in between pages
navigation = st.sidebar.selectbox('Pilih halaman:', ['EDA', 'Predict'])

# make condition
if navigation == 'EDA':
    eda.run()
else:
    predict.run()