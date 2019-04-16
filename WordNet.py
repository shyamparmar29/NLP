# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:50:44 2019

@author: Shyam Parmar
"""

#You can use WordNet alongside the NLTK module to find the meanings of words, synonyms, antonyms, and more.

from nltk.corpus import wordnet

syns = wordnet.synsets("program") # Find synonyms of "program"

print(syns[0].name())  #An example of a synset
print(syns[0].lemmas()[0].name())   #Just the word
print(syns[0].definition())  #Definition of that first synset
print(syns[0].examples())  #Examples of the word in use

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

# we can also easily use WordNet to compare the similarity of two words and their tenses

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))