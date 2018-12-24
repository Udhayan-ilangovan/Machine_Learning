# Natural Language Processing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the dataset
nlp_dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t",quoting = 3)

# Cleaning the text && using NLTK removing unwanted words
import re
#import nltk
# nltk.download("stopwords")
# nltk.download('popular') # download all nltk package
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
port_stem = PorterStemmer()
stopwords_include = ["no","not"]
corpus = []
for i in range (0,1000):    
    review_in =nlp_dataset['Review'][i]
    review_onle_words = re.sub('[^a-zA-Z]',' ',review_in )
    review_low_case = review_onle_words.lower()
    review_list = review_low_case.split()
    r_l_l =len(review_list) 
    review_clean = []
    
    for i in range(0,r_l_l):
        word = review_list[i]
        if ( word  not in  set(stopwords.words("english"))):
            stemed_word = port_stem.stem(word)# using NLTK steming the words EX: loved -> love
            review_clean.append(stemed_word)
        elif ( word in stopwords_include):# Including know important word
            stemed_word = port_stem.stem(word)# using NLTK steming the words EX: loved -> love
            review_clean.append(stemed_word)
        #It is same as the above loop
        #review = [port_stem.stem(word) for word in review_list if not word in set(stopwords.words('english'))]
        # Join the review list as string && finnaly cleaned review
    review_cleaned = " ".join(review_clean)
    corpus.append(review_cleaned)

# while using stopwords this negations are removed. We can combine these words with words around them to create n-grams and analyze those in our model
    # n-gram example >> an apple a day >> 1-gram -> " an" ,"apple", "a", "day" || 2-gram -> "an apple" ,"a day" || 3-gram -> "an apple a" "day"
    
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_features=1500)
X = count_vect.fit_transform(corpus).toarray()
y = nlp_dataset.iloc[:,-1].values
    
# Naive Bayes is one of best algorithm for Natural Language Processing
# Naive Bayes

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state = 0)

# Training the model 
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred_test = nb_classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_test)
accurcy = ((cm[0,0]+cm[1,1])/len(x_test))*100

corpus_re_in = []
def cleaning_input(review_i):
    review_in = review_i
    review_onle_words = re.sub('[^a-zA-Z]',' ',review_in )
    review_low_case = review_onle_words.lower()
    review_list = review_low_case.split()
    r_l_l =len(review_list) 
    review_clean = []
    
    for i in range(0,r_l_l):
        word = review_list[i]
        if ( word  not in  set(stopwords.words("english"))):
            stemed_word = port_stem.stem(word)# using NLTK steming the words EX: loved -> love
            review_clean.append(stemed_word)
        elif ( word in stopwords_include):# Including know important word
            stemed_word = port_stem.stem(word)# using NLTK steming the words EX: loved -> love
            review_clean.append(stemed_word)
        #It is same as the above loop
        #review = [port_stem.stem(word) for word in review_list if not word in set(stopwords.words('english'))]
        # Join the review list as string && finnaly cleaned review
    review_cleaned = " ".join(review_clean)
    corpus_re_in.append(review_cleaned)
    return review_cleaned;

def count_vect_in(review_i):
    review_i = [review_i]
    X_in = count_vect.transform(review_i).toarray()
    return X_in



def predicting( review_i):
    y_pred_in = nb_classifier.predict(review_i)
    return y_pred_in;

def result(review_in):
    review_result = review_input_predict[0]
    print("Thank you for review us")
    if(review_result == 1):
        print("We are thrilled to hear such good feedback, and we’re proud to be one of the coziest restaruent in paris.")
    
    else:
        print("We are sorry to hear about your bad experience.")
        print("We’re normally known for our exceptional attention to detail, and we regret that we missed the mark.")
    
# Realtime prediction
while (True):
    review_input = input("Please enter your review: ")
    review_input_cleaned = cleaning_input(review_input)
    review_input_count_vect = count_vect_in(review_input_cleaned)
    review_input_predict = predicting(review_input_count_vect)
    result(review_input_predict)
    exit_input = input("Press ""q"" To exit: ")
    if (exit_input is "q"):
        break;

    

    
