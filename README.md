# K_NN-Condensed_1_NN
Letter recognition using basic k-NN classification and the condensed 1-NN algorithm on the Letter Recognition Data Set from the UCI Machine Learning Repository ,

Download the Letter Recognition Data Set from the UCI Machine Learning Repository. This dataset contains 20,000 examples. Divide the set so that the first 15,000 examples are for training and the remaining 5,000 for testing.

Implement basic k-NN classification and the condensed 1-NN algorithm described in the course slides. Let nTrain = number of training examples, nTest = number of testing examples, and D be the dimensionality of the examples. You should implement the following functions. (Implementations that do not conform to these specifications will lose a significant amount of credit for this assignment.)

testY = testknn(trainX, trainY, testX, k)

where trainX is a (nTrain *D) data matrix, testX is a (nTest * D) data matrix, trainY is a (nTrain * 1) label vector, and testY is a (nTest * 1) label vector, and k is the number of nearest neighbors for classification.

condensedIdx = condensedata(trainX, trainY)

where condensedIdx is a vector of indicies of the condensed training set.

For both versions of k-NN, run the following experiments

Classification for k = {1,3,5,7,9}

Randomly subsample the training data for N = {100, 1000, 2000, 5000, 10000, 15000}

Notes:

You should run 2 (algorithms) * 5 (values of k) * 6(values of N) = 60 total experiments

"Matrix" == numpy array
