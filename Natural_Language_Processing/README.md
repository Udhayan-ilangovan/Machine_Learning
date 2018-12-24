# Natural language processing
* Natural language processing (NLP) is the branch of computer science and particularly artificial intelligence.
* That deals with the interaction between human languages and computing machine. 
* Especially computers are programmed to recognise, process, and generate language just like a person.
* Tokenization
    * It is generally, an initial process in the NLP, which splits into smaller parts or tokens. 
    * Larger paragraphs can be tokenized into sentences and sentences can be tokenized into words.
* Stop words
    * These are the words which are filtered out before processing of text 
    * These words provide very fewer sense to the overall meaning,  For example
        *  The, a, and
* Parts-of-speech Tagging
    * POS tagging consists of selecting a category tag to a tokenized sentence like nouns, verbs, adjectives.
* Normalization
    * Normalization is generally transforming text to the same case (upper || lower), excluding the punctuations, converting numbers to their word equivalents and etc. 
    * Normalization allows processing to progress uniformly.
* Stemming
    * Stemming is the process of eliminating affixes from a word.
    * LOVING is changed to LOVE
* Lemmatization
    * Lemmatization is similar to stemming, lemmatization is able to catch canonical patterns based on a word's lemma.
    * BETTER is changed as GOOD
* Corpus
    * Corpus means a collection of texts. 
    * Such collections may be formed of a single language of texts or can span multiple languages called corpus. It can be a collection of themed texts like historical, Biblical, etc. 
    * It is commonly used as a variable name to store processed text in NLP.
* Bag of Words
    * It is used to simplify the contents of a selection of text. 
    * It ignores word's order or grammar but it focuses on the number of occurrences of words within the text. 
* In our example 
    * We will process the review of the restaurants and train a classifier model to classify between negative and positive review.
    * To recognize a review is positive or negative in the future.
    * In this, we are using 1000 reviews


## Natural Language Processing
import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

### Importing the dataset
nlp_dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t",quoting = 3)

### Cleaning the text && using NLTK removing unwanted words
import re
###import nltk
'#' nltk.download("stopwords")
'# nltk.download('popular') # download all nltk package
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

### while using stopwords the negations are removed. We can combine these words with words around them to create n-grams and analyze those in our model
    ### n-gram example >> an apple a day >> 1-gram -> " an" ,"apple", "a", "day" || 2-gram -> "an apple" ,"a day" || 3-gram -> "an apple a" "day"
    
### Creating the bag of words model

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_features=1500)

X = count_vect.fit_transform(corpus).toarray()

y = nlp_dataset.iloc[:,-1].values
    
### Naive Bayes is one of best algorithm for Natural Language Processing
### Naive Bayes

### Splitting the dataset into the Training set and Test set
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
    
### Realtime prediction
while (True):

    review_input = input("Please enter your review: ")
    review_input_cleaned = cleaning_input(review_input)
    review_input_count_vect = count_vect_in(review_input_cleaned)
    review_input_predict = predicting(review_input_count_vect)
    result(review_input_predict)
    exit_input = input("Press ""q"" To exit: ")
    if (exit_input is "q"):
        break;

* Sample Dataset

<img width="1439" alt="nlp_sample _dataset" src="https://user-images.githubusercontent.com/32480274/50404378-6eb80f80-07a7-11e9-856d-5caf1f64f88c.png">
￼
* The accuracy percentage of correct prediction.

<img width="1017" alt="nlp_accuracy" src="https://user-images.githubusercontent.com/32480274/50404381-72e42d00-07a7-11e9-9bbf-29fff1a20304.png">
￼
* Prediction result.

    * Predicted using confusion matrix.
    * 0 - 0 => Correct prediction of the negative review.
    * 1 - 1 => Correct prediction of the positive review.
    * 0 - 1 => Incorrect prediction of the negative review.
    * 1 - 0 => Incorrect prediction of the positive review.

<img width="213" alt="nlp_cm" src="https://user-images.githubusercontent.com/32480274/50404383-7677b400-07a7-11e9-89ba-60121a0b9f22.png">
￼
* Apart from testing with the dataset here we can give a review in realtime and get an immediate response based on the given review.

* Larger the dataset higher the prediction accuracy.

<img width="653" alt="nlp_realtim_pred" src="https://user-images.githubusercontent.com/32480274/50404387-7b3c6800-07a7-11e9-8b44-1d4a6da241a0.png">
￼

