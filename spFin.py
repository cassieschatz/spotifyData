#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 00:33:40 2021

@author: cassieschatz
"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def printPlot(var, x, y, m, b):
    plt.title("Relationship between Popularity and " + var)
    m = linReg.coef_
    b = linReg.intercept_
    #plt.xlabel("Salary, Formula: p(x) = " + str(m) + "x + " + str(b))
    plt.ylabel(var)
    plt.scatter(x,y, color='red')
   # plt.plot(x, linReg.predict(x),color='blue')
    plt.show()
    plt.clf()

    
def cleanRawData():
    #Make sure that there are no duplicates
    current = pd.read_csv("rawData.csv")
    current.drop(['num', 'all_artists'], axis = 1)
    current = pd.DataFrame.drop_duplicates(current)
    print(current.shape)
    return current

def betterCM(cm):
    correct = []
    wrong = []
    
    l = len(cm)
    for j in range(l):
        correct.append(cm[j][j])
    
    
    currSum = 0
    
    
    for i in range(l):
        currSum = 0
        for j in range(l):
            if(i != j):
                currSum = cm[i][j] + currSum
        wrong.append(currSum)
    
        
    print("My modiied ")
    print(correct)
    print(wrong)
    
    
    

def decisionTree(X_train, y_train, X_test, y_test):
    #Method 1: Decision Tree
    print("Classifying data...")
    #Classify Data
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    print("Predicting data...")
    #Predict data
    y_pred = classifier.predict(X_test)
    
    print("Decision Tree, CM:")
    betterCM(confusion_matrix(y_test, y_pred))
    print(classifier.score(X_test,y_test))
    
    
    

    #Conclusion: the decision tree came back with a very low score!!!
    #Rounding the values makes it more accurate though (0.06 vs 0.1)
    #Dropping mode added to the accuracy
    #Dropping time_signature took away from the accuracy
    
def knn(X_train, y_train, X_test, y_test):
    print("Classifying data...")
    knn = KNeighborsClassifier(n_neighbors = 2)
    knn.fit(X_train, y_train)

    print("Predicting data...")
    #Predict data
    pred = knn.predict(X_test)
    


    print("Confuson Matrix, KNN:")
    conKNN = confusion_matrix(y_test, pred)
    
    print(betterCM(conKNN))
    #print(classification_report(y_test, pred))
    print(knn.score(X_test,y_test))


#def kmeans():
    
              
             
    

current = cleanRawData()

#Get the popularity values seperate:
x = current['popularity']

#Delete irrelevant features
temp = current.drop(['popularity', 'title', 'first_artist', 'all_artists', 'num', 'id'], axis=1)

#Use linear regression tactics to see if there are any factors which don't have any relationship with popularity:
labels = ['danceability', 'energy', 'key', 'loudness', 'mode', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
labels2 = ['danceability', 'energy', 'key', 'loudness', 'mode', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']


#This was the plotting part
xNew = x.values.reshape(len(x),1)


for var in labels:
    #Initialize the varables
    y = current[var]
    y = y.values.reshape(len(y),1)
    scaler = StandardScaler()
    scaler.fit(y)
    scaled_features = scaler.transform(y)
    scaled_data = pd.DataFrame(y)
    linReg = LinearRegression().fit(xNew,y)
    printPlot(var, x, y, linReg.coef_, linReg.intercept_)


#Scalling the data, making panda array:
scaler = StandardScaler()
scaler.fit(temp)
scaled_features = scaler.transform(temp)
scaled_data = pd.DataFrame(scaled_features, columns = temp.columns)



#Rounding the popularities:
pAll = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ind = 0
for p in x:
    p = (p // 10) * 10
    x.update(pd.Series([p], index = [ind]))
    i = p // 10
    if(i == 10):
        x.update(pd.Series([90], index = [ind]))
        i = 9
    pAll[i] = pAll[i] + 1
    ind = ind + 1


#Rounding the popularites, into two groups:
pAll = [0, 0]
ind = 0
for p in x:
    if(p >= 50):
        x.update(pd.Series([100], index = [ind]))
        pAll[0] = pAll[0] + 1
    else:
        x.update(pd.Series([0], index = [ind]))
        pAll[1] = pAll[1] + 1
    ind = ind + 1

current.drop(['mode'], axis=1)

#Split the data into training/testing:
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(temp, x, test_size=0.40)

pAllPred = [0, 0]
ind = 0
for p in x:
    if(p >= 50):
        x.update(pd.Series([100], index = [ind]))
        pAllPred[0] = pAll[0] + 1
    else:
        x.update(pd.Series([0], index = [ind]))
        pAllPred[1] = pAll[1] + 1
    ind = ind + 1
#print(pAllPred)


knn(X_train, y_train, X_test, y_test)
decisionTree(X_train, y_train, X_test, y_test)
     
