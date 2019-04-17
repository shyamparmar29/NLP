# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:02:34 2019

@author: Shyam Parmar
"""

import nltk
import random
import pickle
from nltk.corpus import movie_reviews

#  In each category (we have pos or neg), take all of the file IDs (each review has its own ID),
# then store the word_tokenized version (a list of words) for the file ID, followed by the positive or 
# negative label in one big list.
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


# We use random to shuffle our documents. This is because we're going to be training and testing.
# If we left them in order, chances are we'd train on all of the negatives, some positives, and 
# then test only against positives. We don't want that, so we shuffle the data.
random.shuffle(documents)


# just so you can see the data you are working with, we print out documents[1], which is a big list, 
# where the first element is a list the words, and the 2nd element is the "pos" or "neg" label.
#print(documents[1])


# We want to collect all words that we find, so we can have a massive list of typical words. 
# From here, we can perform a frequency distribution, to then find out the most common words. 
# As you will see, the most popular "words" are actually things like punctuation, "the," "a" and so on,
# but quickly we get to legitimate words. We intend to store a few thousand of the most popular words, 
# so this shouldn't be a problem.
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())# Convert all words to lower case because casing doesn't matter when classifying a text.

all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))  #gives you the 15 most common words
#print(all_words["stupid"])  #how many occurences a word has

word_features = list(all_words.keys())[:3000]  # word_features, which contains the top 3,000 most common words

#function that will find these top 3,000 words in our positive and negative documents, marking their
# presence as either positive or negative:
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# Saving the feature existence booleans and their respective positive or negative categories by doing
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

# After saving the pickle file
# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)

#save_classifier = open("naivebayes.pickle","wb")  # wb = write in bytes
#pickle.dump(classifier, save_classifier)
#save_classifier.close()