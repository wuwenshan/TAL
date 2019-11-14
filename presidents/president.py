# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:30:42 2019

@author: wuwen
"""

import numpy as np
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin
from sklearn import cross_validation
import sklearn.feature_extraction.text as txt
import matplotlib.pyplot as plt
from math import *
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import f1_score


def readPresidentFile(fileName):
    file = open(fileName, "r", encoding="utf8")
    y = [] # liste des labels (M et C)
    alldocs = [] # liste contenant l'ensemble des documents du corpus d'apprentissage
    for line in file:
        i = line.find('>')
        alldocs.append(line[i+2:len(line)-1])
        y.append(line[i-1:i])
    file.close()
    return alldocs,y

def convertLabelToBinary(y):
    y_bin = []
    for i in range(len(y)):
        if y[i] == 'C':
            y_bin.append(0)
        else:
            y_bin.append(1)
    return y_bin

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def getScoreF1(alldocs,y,testSize,model,isStopWords,isStemming,isLowercase,ngram):
    
    if isStemming:
        alldocs = [[PorterStemmer().stem(word) for word in sentence.split()] for sentence in alldocs]
        alldocs = [" ".join(doc) for doc in alldocs]
        
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(alldocs,y,test_size=testSize,random_state=0)
    
    stopWords = None
    if isStopWords:
        stopWords = stopwords.words('french') 

    ## Vectorisation
    vec = txt.CountVectorizer(token_pattern=r"\b[^\d\W]+\b",stop_words=stopWords,lowercase=isLowercase,ngram_range=ngram) # outil de vectorisation
    bow = vec.fit_transform(X_train) # instatiation de l'outil de vectorisation sur le corpus
    X = bow.tocsr() # passage vers les matrices sparses compatibles avec les modèles statistiques de sklearn
    
    ## Apprentissage
    model.fit(X, y_train)
    
    bowtest = vec.transform(X_test)
    Xtest = bowtest.tocsr()
    ypred = model.predict(Xtest) # usage sur une nouvelle donnée
    
    y_true = convertLabelToBinary(ypred)
    y_pred = convertLabelToBinary(y_test)
    
    return f1_score(y_true, y_pred)

def getTheBestParams(alldocs,y,model):
    ########################################
    ### Param à prendre en compte :      ###
    ###   - Stemming : ( T | F )         ###
    ###   - Stopwords : ( T | F )        ###
    ###   - Lowercase : ( T | F )        ###
    ###   - N-Gram : ( 1 | 2 )           ###
    ########################################
    scores = dict()
    boolean = [True,False]
    i = 0
    for stopword in boolean:
        for stemming in boolean:
            for lowercase in boolean:
                for ngram in [(1,1),(1,2),(2,2)]:
                    for size in [0.4,0.3,0.2,0.1]:
                        score = getScoreF1(alldocs,y,size,model,stopword,stemming,lowercase,ngram)
                        scores[i] = dict()
                        scores[i]['stopword'] = stopword
                        scores[i]['stemming'] = stemming
                        scores[i]['lowercase'] = lowercase
                        scores[i]['ngram'] = ngram
                        scores[i]['size'] = size
                        scores[i]['score'] = score
                        i += 1
    return scores


def getPredict(alldocs,y,alldocstest,model,isStopWords,isStemming,isLowercase,ngram):
    if isStemming:
        alldocs = [[PorterStemmer().stem(word) for word in sentence.split()] for sentence in alldocs]
        alldocs = [" ".join(doc) for doc in alldocs]
    
    stopWords = None
    if isStopWords:
        stopWords = stopwords.words('french') 

    ## Vectorisation
    vec = txt.CountVectorizer(token_pattern=r"\b[^\d\W]+\b",stop_words=stopWords,lowercase=isLowercase,ngram_range=ngram) # outil de vectorisation
    bow = vec.fit_transform(alldocs) # instatiation de l'outil de vectorisation sur le corpus
    X = bow.tocsr() # passage vers les matrices sparses compatibles avec les modèles statistiques de sklearn
    
    ## Apprentissage
    model.fit(X, y)
    
    bowtest = vec.transform(alldocstest)
    Xtest = bowtest.tocsr()
    ypred = model.predict(Xtest) # usage sur une nouvelle donnée
    
    return ypred,X

def readFileTest(fileName):
    file = open(fileName, "r",encoding="utf8")
    alldocstest = [] # liste contenant l'ensemble des documents du corpus d'apprentissage
    for line in file:
        i = line.find('>')
        alldocstest.append(line[i+2:len(line)-1])
    file.close()
    return alldocstest


def plotGraph(xlist,ylist,name):
    fig, ax = plt.subplots()
    for i in range(len(xlist)):
        ax.plot(xlist[i], ylist[i])
    ax.set(xlabel='Pourcentage test', ylabel='Score F1',
           title='Evolution du score F1')
    ax.grid()
    
    fig.savefig(name)
    plt.show()


def plotScores(scores):
    x = []
    y = []
    xlist = []
    ylist = []
    for key,item in scores.items():
        x.append(item['size']*100)
        y.append(item['score'])
        if len(x) == 4 and len(y) == 4:
            xlist.append(x)
            ylist.append(y)
            x = []
            y = []
            if len(xlist) == 3 and len(ylist) == 3:
                s = "fig" + str(key) + ".png"
                plotGraph(xlist,ylist,s)
                xlist = []
                ylist = []

def writeLabels(labels):
    filetest = open("Test.txt", "w")
    for label in labels:
        filetest.write(label + "\n")
    filetest.close()

def postProcessing(ypred):
    # méthode fenêtrée avec les 10 plus proches voisins
    # classes pondérées
    
    # affecter un poids à chaque classe du fait qu'elles sont déséquilibrées
    poids = []

    for i in range(len(ypred)):
        if (ypred[i] == 'C'):
            poids.append(1)
        else:
            poids.append(2.85)
    
    # calcul de la moyenne pondérée   
    res = []     
    for i in range(len(ypred)):
        if (i>4 and i<len(ypred)-5):        
            voisins = list(ypred[i-5:i+5])
            nbC = voisins.count('C')
            nbM = voisins.count('M')
            moy_ponderee = (nbC+(nbM*2.85)) / 11
            res.append(moy_ponderee)        
        else:
            res.append(poids[i])
    
    # lissage
    tab = []
    for i in range(len(res)):
        if (i>4):
            if (res[i] > 1.41):
                tab.append('M')
            else:
                tab.append('C')
        else:
            tab.append(str(ypred[i]))
        
    # on remplace les labels seuls (qui ne sont pas en bloc)
    for i in range(len(res)) :
        if i>0 and i!=len(res)-1:
            if tab[i-1] == tab[i+1] and tab[i] != tab[i-1] :
                if tab[i-1] == 'C':
                    tab[i] = 'C'
                else :
                    tab[i] = 'M'      
    
    return tab

# Calcul de la cross validation
def crossValidation(X,y,model,n):
    scores = cross_validation.cross_val_score( model, X, y, cv=n)
    scoresMean = scores.mean()
    scoresStd = scores.std()
    return scores,scoresMean,scoresStd

# Afficher la cross validation
def plotCV(x,y,std):
    plt.errorbar(x,y,std)
    plt.show()
    return None

# affichage histogramme
def getHist(y):
    fig, ax = plt.subplots()
    model = ('SVM','NB','RL')
    y_pos = np.arange(len(model))
    
    ax.barh(y_pos, y, align='center',
            color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Scores')
    ax.set_title('Moyenne des modèles sur 10-fold CV')
    plt.savefig("moyModels")
    plt.show()

# SVM
clf3 = svm.LinearSVC()
# Naive Bayes
clf2 = nb.MultinomialNB()
# regression logistique
clf = lin.LogisticRegression()

## Lecture corpus
alldocs,y = readPresidentFile("corpus.tache1.learn.utf8")
alldocstest = readFileTest("corpus.tache1.test.utf8")

#scores = getTheBestParams(alldocs,y,clf)

# =============================================================================
# ypred,X = getPredict(alldocs,y,alldocstest,clf,False,False,False,(1,2))
# ypred2, X2 = getPredict(alldocs,y,alldocstest,clf2,False,False,False,(1,2)) 
# ypred,X3 = getPredict(alldocs,y,alldocstest,clf3,False,False,False,(1,2))
# =============================================================================

#score = getScoreF1(alldocs,y,0.1,clf,False,False,False,(1,2))

#plotScores(scores)

#ynew = postProcessing(ypred)

# =============================================================================
# s,sm,std = crossValidation(X,y,clf,10)
# s2,sm2,std2 = crossValidation(X2,y,clf2,10)
# s3,sm3,std3 = crossValidation(X3,y,clf3,10)
# =============================================================================

# =============================================================================
# plt.errorbar(np.arange(1,len(s)+1),s,std)
# plt.errorbar(np.arange(1,len(s2)+1),s2,std2)
# plt.errorbar(np.arange(1,len(s3)+1),s3,std3)
# =============================================================================

# =============================================================================
# moyModel = [sm3,sm2,sm]
# 
# getHist(moyModel)
# =============================================================================
