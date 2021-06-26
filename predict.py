import numpy as np
import csv
import sys
import pickle

from validate import validate
import sys

train_X_file_path = "train_X_re.csv"
train_Y_file_path = "train_Y_re.csv"
validation_split = 0.2

def trainValSplit(X,Y):
    train_X = np.copy(X)
    train_Y = np.copy(Y)
    valIndex = -int(validation_split*(train_X.shape[0]))
    val_X = train_X[valIndex:]
    val_Y = train_Y[valIndex:]
    train_X = train_X[:valIndex]
    train_Y = train_Y[:valIndex]
    return (train_X, train_Y, val_X, val_Y)

def standardize(X, column_indices):
    colMeans = np.nanmean(X, axis=0)
    colSTDs = np.nanstd(X, axis=0)
    for col in column_indices:
        X[:,col] = (X[:,col] - colMeans[col])/colSTDs[col]
    return X

def min_max_normalize(X, column_indices):
    colMins = np.min(X, axis=0)
    colMaxs = np.max(X, axis=0)
    for col in column_indices:
        X[:,col] = (X[:,col]-colMins[col])/(colMaxs[col] - colMins[col])
    return X

def mean_normalize(X, column_indices, colMeans=[], colMins=[], colMaxs=[]):
    if len(colMeans)==0 or len(colMaxs)==0 or len(colMins)==0:
        cols = X[:, column_indices]
        colMeans = np.mean(cols, axis=0)
        colMaxs = np.max(cols, axis=0)
        colMins = np.min(cols, axis=0)
        X[:,column_indices] = (cols-colMeans)/(colMaxs-colMins)
        return X, colMeans, colMins, colMaxs
    else:
        cols = X[:, column_indices]
        X[:,column_indices] = (cols-colMeans)/(colMaxs-colMins)
        return X

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_re.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def getBestSplit(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    bestAttr = -1 
    bestThresh = -1 
    minSSR = np.inf
    for attr in range(len(X[0])):
        for i in range(len(X)-1):
            threshold = (X[i][attr] + X[i+1][attr])/2
            X_left, Y_left, X_right, Y_right = X[X[:,attr] < threshold], Y[X[:,attr] < threshold], X[X[:,attr] >= threshold], Y[X[:,attr] >= threshold]
            SSR = np.sum((Y_left - np.mean(Y_left))**2) + np.sum((Y_right - np.mean(Y_right))**2)
            if SSR < minSSR:
                minSSR = SSR 
                bestAttr = attr 
                bestThresh = threshold
    return (bestAttr, bestThresh)

class Node:
    def __init__(self, depth):
        self.predictedValue = 0
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None

    def setAttrThresh(self, attr, thresh):
        self.feature_index = attr 
        self.threshold = thresh

    def setLeftChild(self, child):
        self.left = child 
        
    def setRightChild(self, child):
        self.right = child
    
    def setPredictedValue(self, val):
        self.predictedValue = val 

def preorder(node):
    if node == None:
        return
    print(f"X{node.feature_index} {node.threshold}")
    preorder(node.left)
    preorder(node.right)

def buildTree(X, Y, depth, maxDepth, minSize):
    if depth >= maxDepth or len(X) < minSize or len(X) == 0:
        return None 
    attr, threshold = getBestSplit(X, Y)
    X_left, Y_left, X_right, Y_right = X[X[:,attr] < threshold], Y[X[:,attr] < threshold], X[X[:,attr] >= threshold], Y[X[:,attr] >= threshold]
    left = buildTree(X_left, Y_left, depth+1, maxDepth, minSize)
    right = buildTree(X_right, Y_right, depth+1, maxDepth, minSize)
    newNode = Node(depth)
    newNode.setAttrThresh(attr, threshold)
    newNode.setLeftChild(left)
    newNode.setRightChild(right)
    if left == None and right == None:
        newNode.setPredictedValue(np.mean(Y))
    elif left == None or right == None:
        if left == None:
            return right 
        else:
            return left
    return newNode

def predictValue(root, X):
    feature = root.feature_index 
    threshold = root.threshold 
    if X[feature] < threshold:
        if root.left != None:
            return predictValue(root.left, X)
        else: 
            return root.predictedValue
    else:
        if root.right != None:
            return predictValue(root.right, X)
        else: 
            return root.predictedValue

def makePredictions(root, X):
    Y = []
    for x in X:
        Y.append(predictValue(root, x))
    return np.array(Y)

def trainModel():
    train_X = np.genfromtxt(train_X_file_path, delimiter=",", skip_header=1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter=",", skip_header=0)
    train_X, train_Y, val_X, val_Y = trainValSplit(train_X, train_Y)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    bestRoot = None 
    minLoss = np.inf
    bestMD = None 
    bestMS = None 
    for maxDepth in range(1,10):
        for minSize in range(1,10):
            root = buildTree(train_X, train_Y, 0, maxDepth, minSize)
            # preorder(root)
            Y = makePredictions(root, val_X)
            loss = np.sum((np.array(Y) - np.array(val_Y))**2)/len(val_Y)
            print(f"maxDepth = {maxDepth}, minSize = {minSize}\n\tVal_Loss = {loss}")
            if loss < minLoss:
                minLoss = loss 
                bestRoot = root 
                bestMD = maxDepth 
                bestMS = minSize
    print(f"bestMD = {bestMD}, bestMS = {bestMS}, minLoss = {minLoss}")
    return bestRoot

def predict(test_X_file_path):

    if "-trainModel" in sys.argv:
        model = trainModel()
        with open("./model_file.sav", "wb") as fp: 
            pickle.dump(model, fp)
    
    test_X = import_data(test_X_file_path)

    # Load Model Parameters
    model = pickle.load(open("./model_file.sav", "rb"))
    
    # Predict Target Variables
    pred_Y = makePredictions(model, test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_re.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_re.csv") 