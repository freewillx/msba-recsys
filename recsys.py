import csv
import numpy as np
import pandas as pd

# Load data files
trainDataFile = './data/toy_train.csv'
testDataFile = './data/toy_test.csv'
# trainDataFile = './data/restaurant_train.csv'
# testDataFile = './data/restaurant_test.csv'

trainData = pd.read_csv(trainDataFile)
trainData.columns = ['user_id', 'item_id', 'rating']

testData = pd.read_csv(testDataFile)
testData.columns = ['user_id', 'item_id', 'rating']

# Compare users and determine if any test users are not in the training data set
uniqueTrainUser = trainData.user_id.unique()
uniqueTestUser = testData.user_id.unique()
uniqueCompareMask = np.in1d(uniqueTestUser, uniqueTrainUser, invert=True)

# Find test users that are not in the training data
testNotInTrain = uniqueTestUser[uniqueCompareMask]
assert len(testNotInTrain) == 0

'''
From the output we know that all test data set users exists in the training data.
To optimize the computation, we will first find the 40 nearest neighbours for each user using the trainning data set
and we calculate the prediction with test data
'''

# Create rating matrix with training data
trainMatrix = trainData.pivot_table(values = 'rating', index = 'user_id', columns = 'item_id')
assert trainMatrix.shape == (len(trainData.user_id.unique()), len(trainData.item_id.unique()))

print trainMatrix.shape

# Calculate average rating for all users
trainUserMean = pd.DataFrame(data=trainMatrix.mean(axis=1), index=trainMatrix.index)

# Define cosine similarity function
def cosineSim(u1Rating, u2Rating):
    u1RatingSqrtSum = np.sum(u1Rating.apply(lambda x: x**2))
    u2RatingSqrtSum = np.sum(u2Rating.apply(lambda x: x ** 2))
    return np.sum(u1Rating * u2Rating) / np.sqrt(u1RatingSqrtSum * u2RatingSqrtSum)

# Find 40 nearest neighbours with cosine sim 2.5 (train)

def calculateUserSimMatrix(mtrx):
    simMatrix = pd.DataFrame()
    for i in range(0, len(mtrx)):
        u = mtrx[mtrx.index == mtrx.index[i]].squeeze()
        simSerie = mtrx.apply(lambda x: cosineSim(u, x.squeeze()), axis =1)
        simMatrix[u.name] = simSerie
    return simMatrix

simMatrix = calculateUserSimMatrix(trainMatrix)

# TODO get user1 top 40
simMatrix.ix['user1'].order(ascending=False)[1:41]









# http://stackoverflow.com/questions/25736861/python-pandas-finding-cosine-similarity-of-two-columns
trainMatrix.ix[1:10,1:10]
trainMatrix.ix['user39', 'item17']
trainMatrix.ix['user12', 'item11']

trainMatrix.ix['user39'] * trainMatrix.ix['user39'] # trainMatrix.ix['user39'] ** 2

# Prediction based on 40 nearest neighbours 2.3 (test)
def predictRating(uer_id, item_id):
    print uer_id + " " + item_id
    return 3
