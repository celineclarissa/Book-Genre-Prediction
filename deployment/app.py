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

Problem Statement and Objective

The process begins with exploratory data analysis (EDA) to uncover patterns in book genres, followed by feature engineering to enhance the
dataset. A predictive model is then built using an Artificial Neural Network (ANN), with ongoing efforts to optimize its performance. The
goal is to achieve an accuracy above 90%, after which the final model will be deployed on Hugging Face within seven working days. The web
application will also include a dedicated page for interactive EDA visualizations.

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