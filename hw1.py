import csv
import random
import sys
import timeit
import urllib2
from collections import Counter
import pandas as pd
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
response = urllib2.urlopen(url)
cr = csv.reader(response)

a = np.genfromtxt(response, dtype=str, delimiter=',')
for i in range(20000):
    p = a[i][0]
    a[i][0] = ord(p)  # converting labels to ascii values since arrays are of same type  of data type

y = a.astype(np.float)  # converting  whole dataset to float type

trainingDataset = y[0:15000, 0:17]
testX = y[15000:, 1:17]  # last 5000 are test data
validateX = y[15000:, 0:1]  # correct classes of test data

sample = int(raw_input('Enter sample of training data '))


def randtrainset(x, N):
    np.random.shuffle(x)
    shortgu = x[0:N]
    return shortgu


def confumatrix(tru, pred):
    print confusion_matrix(tru, pred)


def convecharordarray(char):
    dmmy = np.empty((np.shape(char)[0], 1), dtype=np.int)
    for eav in range(np.shape(char)[0]):
        dmmy[eav, 0] = ord(char[eav, 0])
    return dmmy


def conveordchararray(orde):
    """

    :rtype: object
    """
    dmmay = np.empty((np.shape(orde)[0], 1), dtype=str)
    for eav in range(np.shape(orde)[0]):
        dmmay[eav, 0] = chr(int(orde[eav, 0]))
    return dmmay


def testknn(trainx, traiNy, testx, t):
    """

    :rtype: object
    """
    dummy = np.zeros((int(np.shape(testx)[0]), 1), dtype=np.int)  #
    trainy = convecharordarray(traiNy)

    for idx, tesiter in enumerate(testx):
        respdistances = []
        testiterarr = np.matrix(np.tile(tesiter, (trainx.shape[0], 1)))
        trainxmat = np.matrix(trainx)
        difference = testiterarr - trainxmat
        squar = np.square(difference)
        su = np.sum(squar, axis=1)
        dist = np.sqrt(su)
        fla = dist.flatten().tolist()[0]
        for itf in range(t):
            smal = fla.index(min(fla))
            respdistances.append(smal)
            fla[smal] = sys.float_info
        for inf in range(t):
            respdistances[inf] = trainy[respdistances[inf]][0]

        p = Counter(respdistances)
        v = list(p.values())
        kp = list(p.keys())
        maxm = kp[v.index(max(v))]
        dummy[idx] = maxm
    return conveordchararray(dummy)


def convarray2char(poij):
    polp = []
    for injk in poij:
        polp.append(chr(injk))
    return polp


def accuracychecker(test, validate):
    j = 0
    for iy, every in enumerate(range(5000)):
        if (validate[iy][0]) == (test[iy][0]):
            j += 1

    accurpercent = (float(j * 100) / 5000)
    print accurpercent

    return accurpercent


def condensedata(Trainx, Trainy):
    index = []
    orgtrainx = Trainx
    trainx = Trainx
    trainy = convecharordarray(Trainy)
    intialSeed = random.randint(0, int(trainx.shape[0]))
    inttrainx = trainx[intialSeed:intialSeed + 1, :]
    inttrainy = trainy[intialSeed:intialSeed + 1, :]
    index.append(intialSeed)
    np.delete(trainx, intialSeed, 0)
    np.delete(trainy, intialSeed, 0)
    abc = 1

    inttestY = convecharordarray(testknn(inttrainx, conveordchararray(inttrainy), trainx, 1))

    # while enterloop(inttestY, trainy):
    while np.array_equal(inttestY, trainy) == bool(False):

        detto = 1
        abc += 1
        lis = range(trainx.shape[0])
        shuflis = random.sample(lis, trainx.shape[0])
        for xin in shuflis:

            if inttestY[xin][0] != trainy[xin][0] and detto == 1:
                inttrainx = np.concatenate([inttrainx, trainx[xin:xin + 1, :]])
                inttrainy = np.concatenate([inttrainy, trainy[xin:xin + 1, :]])
                for eachrow, df in enumerate(orgtrainx):
                    if np.array_equal(orgtrainx[eachrow:eachrow + 1, :], trainx[xin:xin + 1, :]):
                        index.append(eachrow)

                np.delete(trainx, xin, 0)
                np.delete(trainy, xin, 0)
                detto += 1

        # ram = int (random.choice(wrong))  wrong.append(xin)
        inttestY = convecharordarray(testknn(inttrainx, conveordchararray(inttrainy), trainx, 1))

    condtest = convecharordarray(testknn(trainX, trainY, testX, po))
    print 'the accuracy of  the condensed nearest neighbours is %f ' % accuracychecker(condtest, validateX)

    return index


short = randtrainset(trainingDataset, sample)

trainX = short[:, 1:17]
trainY = conveordchararray(short[:, 0:1])
algoSelection = int(raw_input('for basic algorithm enter 1 , condensed algorithm 2'))

if algoSelection == 2:
    start = timeit.default_timer()
    po = int(raw_input('Enter no of neighbours (k): '))
    condensedIdx = condensedata(trainX, trainY)
    stop = timeit.default_timer()
    print 'time taken is %f minutes ' % ((stop - start) / 60)
    # print condensedIdx
    print len(condensedIdx)
    print condensedIdx

if algoSelection == 1:
    k = int(raw_input('Enter no of neighbours (k): '))

    start = timeit.default_timer()
    testY = testknn(trainX, trainY, testX, k)
    print 'the accuracy of  the k neighbours is %f ' % accuracychecker(convecharordarray(testY), validateX)
    stop = timeit.default_timer()
    print 'time taken is %f minutes ' % ((stop - start) / 60)

    '''
    print 'confusion matrix'
    y_actu = pd.Series(convarray2char(validateX), name='Actual')
    y_pred = pd.Series(convarray2char(testY), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    df_confusion.to_csv(r'C:\Users\ssirigin\PycharmProjects\pandas.txt', header='Confusion Matrix', index=None, sep=' ',
                        mode='a')
'''

