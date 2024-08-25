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
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud

# set page title
st.set_page_config(
    page_title = 'Graded Challenge 7'
)

# make function run()
def run():
    # make title
    st.title('Book Genre Data EDA')

    # make description
    st.write('This page was made to predict book genre.')

    # Membuat garis lurus
    st.markdown('---')

    # Show dataframe
    st.write('### Dataset')
    df = pd.read_csv('data.csv')

    # show dataset
    st.dataframe(df)

    # make border
    st.write('')
    st.markdown('---')
    st.write('')

    # EDA 1
    # make title
    st.write('### Genre Distribution')

    # make visualization
    fig1 = plt.figure(figsize=[15, 5])
    # count value for each genre
    genre_counts = df['genre'].value_counts(dropna=False)
    # define class names
    class_names = ['fantasy', 'science', 'crime', 'history', 'horror', 'thriller', 'psychology', 'romance', 'sports', 'travel']
    # create pie chart
    plt.pie(genre_counts, autopct='%1.1f%%', labels=class_names, shadow=True)
    plt.title('Genre Pie Chart')
    plt.axis('equal')
    # show visualization
    st.pyplot(fig1)

    # show insight for EDA 1
    st.write("From the pie chart above, it can be analyzed that the genre distribution in data isn't too equal. More than half the genres (fantasy, science, crime, history, horror, thriller) have similar shares (around 20%). Meanwhile, the others (psychology, romance, sports, travel) also have similar shares (around 2%).")



    # make border
    st.markdown('---')



    # EDA 2
    # make title
    st.write('### Number of Sentences and Words in Each Genre')

    # count sentences and words in 'summary' column
    df['sentence_count'] = df['summary'].apply(lambda x: len(nltk.sent_tokenize(x)))
    df['word_count'] = df['summary'].apply(lambda x: len(nltk.word_tokenize(x)))
    # user input
    choice_eda2 = st.selectbox('Choose genre:', ['fantasy', 'science', 'crime', 'history', 'horror', 'thriller', 'psychology', 'romance', 'sports', 'travel'])
    # create histogram for number of sentences
    fig2_1 = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.histplot(df[df['genre'] == choice_eda2]['sentence_count'], kde=True, bins = 30)
    plt.title(f'Number of Sentences in Genre {choice_eda2}')
    plt.show()
    # create histogram for number of words
    fig2_2 = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.histplot(df[df['genre'] == choice_eda2]['word_count'], kde=True, bins = 30)
    plt.title(f'Number of Words in Genre {choice_eda2}')
    plt.show()
    # show visualization
    st.pyplot(fig2_1)
    st.pyplot(fig2_2)

    # show insight for EDA 2
    st.write("From the bar plots above, it can be understood that the 'number of sentences' and 'number of words' plots are quite similar in each genre. It can also be analyzed that genres 'fantasy', 'science', 'crime', 'history', 'horror', and 'thriller' have relatively low of number of sentences and number of words. Meanwhile, genres ' psychology', 'romance', 'sports', and 'travel' have relatively higher number of sentences and number of words.")



    # make border
    st.markdown('---')



    # EDA 3
    # make title
    st.write('### Wordcloud for Each Genre')
    # user input
    choice_eda3 = st.selectbox('Choose genre:    ', ['fantasy', 'science', 'crime', 'history', 'horror', 'thriller', 'psychology', 'romance', 'sports', 'travel'])
    # create wordcloud
    fig3 = plt.figure(figsize = (20,20))
    wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.genre == choice_eda3].summary))
    plt.title(f'Wordcloud for Genre {choice_eda3}')
    plt.imshow(wc , interpolation = 'bilinear')
    # show visualization
    st.pyplot(fig3)
    # show insight for EDA 3
    st.write("")

# execute file
if __name__=='__main__':
    run()