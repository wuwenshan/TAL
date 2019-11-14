#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:04:38 2019

@author: 3874034
"""

from os import listdir
from os.path import isfile, join

import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction.text as txt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import csv

# SVM
clf = svm.LinearSVC()
# Naive Bayes
#clf = nb.MultinomialNB()
# regression logistique
#clf = lin.LogisticRegression()

# retourne un tableau contenant les noms des fichiers d'un dossier
# path : dossier
def getFilesFromFolder(path):
    fichiers = [f for f in listdir(path) if isfile(join(path, f))]
    return fichiers

def setContainFromFiles(fichiers,label,contenu):
    for i in range(len(fichiers)):
        if (label == 0):
            fichier = open("movies1000/neg/"+fichiers[i], "r", encoding="utf-8")
        else:
            fichier = open("movies1000/pos/"+fichiers[i], "r", encoding="utf-8")
            
        contenu.append((label,fichier.read()))
        """
        for line in fichier:
            newLine = line.strip().split('\n')
            contenu.append((label,newLine))
        """
        fichier.close()

def makeNltkStopWords(languages=['french', 'english', 'german', 'spanish']):
    stop_words = []
    for l in languages:
        for w in stopwords.words(l):
           stop_words.append(w.encode('utf-8')) #w.decode('utf-8') buggait... avec certains caractères
    return stop_words

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def getPredictionCV(alldocs,y,codage,model,isStopWords,isStemming,isLowercase,ngram,n):
    
    vec = None
    
    if isStemming:
        alldocs = [[PorterStemmer().stem(word) for word in sentence.split()] for sentence in alldocs]
        alldocs = [" ".join(doc) for doc in alldocs]
    
    stopWords = None
    if isStopWords:
        stopWords = stopwords.words('english') 

    ## Vectorisation
    if codage == "TFIDF":
        vec = TfidfVectorizer(token_pattern=r"\b[^\d\W]+\b",stop_words=stopWords,lowercase=isLowercase,ngram_range=ngram)
    elif codage == "TF":
        vec = txt.CountVectorizer(token_pattern=r"\b[^\d\W]+\b",stop_words=stopWords,lowercase=isLowercase,ngram_range=ngram) # outil de vectorisation
    elif codage == "Binary":
        vec = txt.CountVectorizer(token_pattern=r"\b[^\d\W]+\b",stop_words=stopWords,lowercase=isLowercase,ngram_range=ngram,binary=True)
        
    bow = vec.fit_transform(alldocs) # instatiation de l'outil de vectorisation sur le corpus
    X = bow.tocsr() # passage vers les matrices sparses compatibles avec les modèles statistiques de sklearn
    
    scores = cross_validation.cross_val_score( model, X, y, cv=n)
    
    return scores

def getTheBestParams(alldocs,y):
    ########################################
    ### Param à prendre en compte :      ###
    ###   - Stemming : ( T | F )         ###
    ###   - Stopwords : ( T | F )        ###
    ###   - Lowercase : ( T | F )        ###
    ###   - N-Gram : ( 1 | 2 )           ###
    ########################################
    # SVM
    clfSVM = svm.LinearSVC()
    # Naive Bayes
    clfNB = nb.MultinomialNB()
    # regression logistique
    clfLR = lin.LogisticRegression()
    scores = dict()
    boolean = [True,False]
    i = 0
    
    for codage in ['TFIDF','TF','Binary']:
        for model in [clfSVM,clfNB,clfLR]:
            for stopword in boolean:
                for stemming in boolean:
                    for lowercase in boolean:
                        for ngram in [(1,1),(1,2),(2,2)]:
                            print("How long have you been here ? ")
                            print("I've been here for ",i+1)
                            score = getPredictionCV(alldocs,y,codage,model,stopword,stemming,lowercase,ngram,5)
                            scores[i] = dict()
                            scores[i]['stopword'] = stopword
                            scores[i]['stemming'] = stemming
                            scores[i]['lowercase'] = lowercase
                            scores[i]['ngram'] = ngram
                            scores[i]['codage'] = codage
                            scores[i]['model'] = model
                            scores[i]['score'] = score
                            i += 1
    return scores


# Param svm_str : classe svm convertie en String
def writeScoresSVMtoCSV(scores,svm_str):
    
    
    with open('scores_svm.csv', mode='w') as csv_file:
        fieldnames = ['Codage', 'Stopwords', 'Stemming','Lowercase','N-Grams','SVM']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key,item in scores.items():
            if str(item['model']) == svm_str:
                writer.writerow({'Codage':item['codage'],'Stopwords':item['stopword'],'Stemming':item['stemming'],'Lowercase':item['lowercase'],'N-Grams':item['ngram'],'SVM':item['meanCV']})


# Param nb_str : classe nb convertie en String
def writeScoresNBtoCSV(scores,nb_str):
    
    
    with open('scores_nb.csv', mode='w') as csv_file:
        fieldnames = ['Codage', 'Stopwords', 'Stemming','Lowercase','N-Grams','NB (CV 5-fold)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key,item in scores.items():
            if str(item['model']) == nb_str:
                writer.writerow({'Codage':item['codage'],'Stopwords':item['stopword'],'Stemming':item['stemming'],'Lowercase':item['lowercase'],'N-Grams':item['ngram'],'NB (CV 5-fold)':item['meanCV']})

# Param lr_str : classe lr convertie en String
def writeScoresLRtoCSV(scores,lr_str):
    
    
    with open('scores_lr.csv', mode='w') as csv_file:
        fieldnames = ['Codage', 'Stopwords', 'Stemming','Lowercase','N-Grams','LR (CV 5-fold)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key,item in scores.items():
            if str(item['model']) == lr_str:
                writer.writerow({'Codage':item['codage'],'Stopwords':item['stopword'],'Stemming':item['stemming'],'Lowercase':item['lowercase'],'N-Grams':item['ngram'],'LR (CV 5-fold)':item['meanCV']})



pathNeg = "movies1000/neg"
pathPos = "movies1000/pos"

fichiersNeg = getFilesFromFolder(pathNeg)
fichiersPos = getFilesFromFolder(pathPos)

contenu = []

setContainFromFiles(fichiersNeg,0,contenu)
setContainFromFiles(fichiersPos,1,contenu)

alldocs = [text for label,text in contenu]
y = [label for label,text in contenu]

#scores = getTheBestParams(alldocs,y)

# =============================================================================
# bestavg = scores[0]['meanCV']
# for i in range(1,len(scores)):
#     if scores[i]['meanCV']>bestavg:
#         bestavg = scores[i]['meanCV']
#         
#     
# =============================================================================
# =============================================================================
# svm_str = str(scores[10]['model'])
# nb_str = str(scores[100]['model'])
# lr_str = str(scores[50]['model'])
# 
# writeScoresSVMtoCSV(scores)
# writeScoresNBtoCSV(scores)
# writeScoresLRtoCSV(scores)
# =============================================================================
# =============================================================================
# testSentiment = open("testSentiment.txt", "r", encoding="utf-8")
# 
# avis = testSentiment.read().split('\n')
# # new_avis = [s.replace('<br /><br />',"") for s in avis]
#     
# testSentiment.close()
# =============================================================================

