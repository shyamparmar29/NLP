# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:17:44 2019

@author: Shyam Parmar
"""

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))

'''A very similar operation to stemming is called lemmatizing. The major difference between these is,
stemming can often create non-existent words, whereas lemmas are actual words.

So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary,
but you can look up a lemma.

Some times you will wind up with a very similar word, but sometimes, you will wind up with a completely 
different word. '''