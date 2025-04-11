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