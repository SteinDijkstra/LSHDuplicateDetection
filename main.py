import json
import re
import numpy as np
import random
from random import shuffle
import time
from sklearn.cluster import AgglomerativeClustering
import sys

# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform

# Comparison made: LSH - total possible pairs LSH
# 0.02 f1* is quite good
# Complete linkage is a good alternative
#


locationFile = "TVs-all-merged.json"


def main(rString):
    print(rString)
    r=int(rString)

    n = 360
    assert (n % r == 0)
    b = int(n/r)

    length = 4
    numberBootstraps=5
    percentageTrainTest=0.63

    data = importData()
    print(len(data))
    simpleData = simplifyData(data)
    print(numberOfPairs(simpleData))
    bootstraps=bootstrap(simpleData,numberBootstraps,percentageTrainTest)
    # #
    results = np.zeros((numberBootstraps, 7))
    times=np.zeros((numberBootstraps,1))
    for iteration,(trainSet, testSet) in enumerate(bootstraps):
        print(f"starting with bootstrap {iteration}")
        startCycle = time.perf_counter()
        threshold= determineThreshold(trainSet, length, b, r)/20
        results[iteration] = run(trainSet, length, b, r, threshold)
        endCycle = time.perf_counter()
        times[iteration]=endCycle-startCycle

    #print(f"average time per bootstrap {np.mean(times)}")
    mean=np.mean(results,axis=0)
    for result in mean:
        print(result)
    print(np.mean(times))
    return


def determineThreshold(data,length,b,r):
    shingleMatrix = createShingleMatrix(data, length)
    signatureMatrix = createSignatureMatrix(shingleMatrix, b * r)
    pairsLSH = lsh(signatureMatrix, b, r)
    distanceMatrix = createDissimilarityMatrix(pairsLSH, data, signatureMatrix, length)
    F1Best=0
    threshold=1
    for t in range(0,21):
        pairsCluster = clusterMethod(distanceMatrix, t/20)
        potentialPairsCluster = clusterMethod(distanceMatrix, t/20)
        [_, _, _, _, _, F1,_]=evaluateResultsCluster(potentialPairsCluster, potentialPairsCluster, data)
        if(F1>F1Best):
            threshold=t
            F1Best=F1
    return threshold

def runLSH(data,length,b,r,t):
    shingleMatrix = createShingleMatrix(data, length)
    signatureMatrix = createSignatureMatrix(shingleMatrix, b * r)
    pairsLSH = lsh(signatureMatrix, b, r)
    return evaluateResults(pairsLSH,data)


def run(data,length,b,r,threshold):
    shingleMatrix = createShingleMatrix(data, length)
    signatureMatrix = createSignatureMatrix(shingleMatrix, b * r)
    pairsLSH = lsh(signatureMatrix, b, r)
    dissimilarityMatrix = createDissimilarityMatrix(pairsLSH, data, signatureMatrix, length)
    pairsCluster = clusterMethod(dissimilarityMatrix, threshold)
    return evaluateResultsCluster(pairsLSH, pairsCluster, data)


def timemethods():
    data = importData()
    simpleData = simplifyData(data)
    startShingle = time.perf_counter()
    shingleMatrix = createShingleMatrix(simpleData, 3)
    endShingle = time.perf_counter()
    print(f"shingling takes {endShingle - startShingle}")
    startSignature = time.perf_counter()
    signatureMatrix = createSignatureMatrix(shingleMatrix, 800)
    endSignature = time.perf_counter()
    print(f"signature takes {endSignature - startSignature}")
    startLSH = time.perf_counter()
    potentialpairs = lsh(signatureMatrix, 80, 10)
    endLSH = time.perf_counter()
    print(f"lsh takes {endLSH - startLSH}")


def importData():
    data = json.load(open(locationFile))
    return data


def simplifyData(data):
    simpleData = []
    for k, v in data.items():
        for item in v:
            #getModelwords(item)
            #print(item)
            simpleData.append((k, cleanString(item.get('title')),cleanFeatures(item),item.get("shop")))
    return simpleData



def cleanString(title):
    title = title.lower()  # remove capital letters
    # replace unit (Inch)
    title = title.replace("\"", "inch")
    title = title.replace("inches", "inch")
    # replace unit (Hertz)
    title = title.replace("hertz", "hz")
    # remove stopwords (maybe)
    title = title.replace("best buy", "")
    title = title.replace("newegg.com","")
    # remove non-alphanumeric characters
    pattern = re.compile('[^\sa-zA-z0-9.]+')
    title = pattern.sub('', title)

    return title

def cleanFeatures(keyvaluePairs):
    keyvaluePairs=keyvaluePairs['featuresMap']
    for k, v in keyvaluePairs.items():
        if k != "title":
            v=cleanString(v)
            keyvaluePairs[k]=v
    return keyvaluePairs

def bootstrap(data, amount, percentage):
    N=len(data)
    nSplit=int(N*percentage)
    trainSets = []
    testSets = []
    for iteration in range(amount):
        shuffle(data)
        train= data[:nSplit]
        test=data[nSplit:]
        trainSets.append(train)
        testSets.append(test)
    return zip(trainSets, testSets)


def shingles(text, length):
    return [text[i:i + length] for i in range(len(text) - length + 1)]


def getModelWordsTitle(title):
    patternmwTitle = re.compile(r'(\b([0-9]+[0-9.]*)[a-z]+\b)|(\b[a-z.]+\b)')
    mwTitle = [x for sublist in patternmwTitle.findall(title)for x in sublist if x != ""]
    return mwTitle


def getModelWords(title,keyvaluePairs):
    mwTitle=getModelWordsTitle(title)
    # mwKeyValuePair=[]
    # #print(keyvaluePairs)
    # for k,v in keyvaluePairs.items():
    #     patternmwkeyvalue=re.compile(r'(\b[0-9]+[0-9.]*[a-z]+\b)')
    #     mwKeyValuePair.extend([x for sublist in patternmwkeyvalue.findall(v) for x in sublist if x != ""])
    #     #[x for sublist in patternmwkeyvalue.findall(v) for x in sublist if x != ""]
    # #print(mwKeyValuePair)
    # mwTitle.extend(mwKeyValuePair)
    return mwTitle


def shingleSet(simpleData, length):
    shSet = set()
    for pair in simpleData:
        mw=getModelWords(pair[1], pair[2])
        shSet.update(mw)
    return shSet


def createShingleMatrix(simpleData, length):
    # might be optimized
    nItems = len(simpleData)
    completeShingle = list(shingleSet(simpleData, length))
    nShingles = len(completeShingle)
    matrix = np.zeros((nItems, nShingles))
    for iterItem, item in enumerate(simpleData):
        shinglesOfTitle = getModelWords(item[1], item[2])
        matrix[iterItem] = np.isin(completeShingle, shinglesOfTitle)
    return matrix


def createSignatureMatrix(shingleMatrix, permutations):
    nItems, nShingles = shingleMatrix.shape
    shingleMatrix[shingleMatrix == 0] = np.nan
    result = np.zeros((permutations, nItems), dtype=int)
    hashObj = list(range(0, nShingles))
    for perm in range(permutations):
        shuffle(hashObj)
        result[perm] = np.transpose(np.nanargmin(shingleMatrix * hashObj, axis=1))
    return result


def lsh(signatureMatrix, b, r):
    nHash, nItems = signatureMatrix.shape
    assert (nHash % b == 0)
    assert (b * r == nHash)
    bands = np.split(signatureMatrix, b, axis=0)
    potentialPairs = set()
    for band in bands:
        hashDict = {}
        for item, column in enumerate(band.transpose()):
            hashColumn = column.tobytes()
            # print(hashColumn)
            if hashColumn in hashDict:
                # print(hashDict[hashColumn])
                hashDict[hashColumn] = np.append(hashDict[hashColumn], item)
            else:
                hashDict[hashColumn] = np.array([item])
        for potentialPair in hashDict.values():
            if len(potentialPair) > 1:
                for i, item1 in enumerate(potentialPair):
                    for j in range(i + 1, len(potentialPair)):
                        if (potentialPair[i] < potentialPair[j]):
                            potentialPairs.add((potentialPair[i], potentialPair[j]))
                        else:
                            potentialPairs.add((potentialPair[j], potentialPair[i]))

    return potentialPairs


def createDissimilarityMatrix(potentialPairs, simpleData, shingleMatrix, length):
    nItems = len(simpleData)
    result = np.full((nItems, nItems), 100, dtype=float)
    np.fill_diagonal(result, 0)


    for pair in potentialPairs:
        item1 = pair[0]
        item2 = pair[1]

        result[item1, item2] = 1 - jaccardDistance(set(shingles(simpleData[item1][1],length)),set(shingles(simpleData[item2][1],length)))
        if(simpleData[item1][3]==simpleData[item2][3]):
            result[item1, item2]=100
        if "Brand" in simpleData[item1][2] and "Brand" in simpleData[item2][2]:
            if (simpleData[item1][2].get("Brand") == simpleData[item2][2].get("Brand")):
                result[item1, item2] = 100
        #result[item1, item2] = 1- cosDistance(shingleMatrix[item1],shingleMatrix[item2])
        #result[item1,item2]= jaccardDistance2(shingleMatrix[item1],shingleMatrix[item2])
        #jaccardDistance(set(getModelWords(simpleData[item1][1], simpleData[item1][2])),
        #                set(getModelWords(simpleData[item2][1], simpleData[item2][2])))

        #if(simpleData[item1][0]==simpleData[item2][0]):
        result[item2, item1] = result[item1, item2]
    return result


def jaccardDistance(shWord1, shWord2):
    return len(shWord1.intersection(shWord2)) / len(shWord1.union(shWord2))

#def jaccardDistance2(vector1,vector2):
#    equal=np.sum(vector1[vector1==vector2])
#    unequal=np.sum(vector1!=vector2)
#    return equal/(unequal+equal)

# def cosDistance(shingleMatrix1,shingleMatrix2):
#    return np.dot(shingleMatrix1,np.transpose(shingleMatrix2))/(np.linalg.norm(shingleMatrix1)*np.linalg.norm(shingleMatrix2))


def clusterMethod(similarityMatrix, t):
    linkage = AgglomerativeClustering(n_clusters=None, affinity="precomputed",
                                      linkage='average', distance_threshold=t)
    clusters = linkage.fit_predict(similarityMatrix)
    dictCluster = {}
    for index, clusternr in enumerate(clusters):
        if clusternr in dictCluster:
            dictCluster[clusternr] = np.append(dictCluster[clusternr], index)
        else:
            dictCluster[clusternr] = np.array([index])
    potentialPairs = set()
    for potentialPair in dictCluster.values():
        if len(potentialPair) > 1:
            for i, item1 in enumerate(potentialPair):
                for j in range(i + 1, len(potentialPair)):
                    if (potentialPair[i] < potentialPair[j]):
                        potentialPairs.add((potentialPair[i], potentialPair[j]))
                    else:
                        potentialPairs.add((potentialPair[j], potentialPair[i]))
    return potentialPairs


def evaluateResultsCluster(potentialPairsLSH,potentialPairsCluster, data):
    tpLSH=0
    tpCluster=0
    for pair in potentialPairsLSH:
        if data[pair[0]][0] == data[pair[1]][0]:
            tpLSH = tpLSH + 1
    for pair in potentialPairsCluster:
        if data[pair[0]][0] == data[pair[1]][0]:
            tpCluster = tpCluster + 1
    numberDuplicates = numberOfPairs(data)
    numberCandidatesLSH = len(potentialPairsLSH) +0.000001
    tpandfn=len(potentialPairsCluster)+0.000001
    PC = tpLSH / numberDuplicates
    PQ = tpLSH / numberCandidatesLSH
    F1StarLSH = 2*(PQ*PC)/(PQ+PC +0.000001)

    precision = tpCluster / numberDuplicates
    recall = tpCluster / tpandfn
    F1 = 2 * (precision * recall)/(precision + recall +0.000001)
    N=len(data)
    totalComparisons=(N*(N-1))/2
    fraction =numberCandidatesLSH/totalComparisons
    #print(f"PC LSH: {PC} PQ LSH: {PQ} F1star LSH: {F1StarLSH}")
    #print(f"precision Cluster:{precision} Recall Cluster:{recall} F1 Cluster: {F1}")
    return [PC, PQ, F1StarLSH, precision, recall, F1, fraction]


def evaluateResults(potentialPairsLSH, data):
    tpLSH = 0
    for pair in potentialPairsLSH:
        if data[pair[0]][0] == data[pair[1]][0]:
            tpLSH = tpLSH + 1
    numberDuplicates = numberOfPairs(data)+0.0000001
    numberCandidates = len(potentialPairsLSH)+0.0000001
    PC = tpLSH / numberDuplicates
    PQ = tpLSH / numberCandidates
    F1StarLSH = 2*(PQ*PC)/(PQ+PC+0.0000001)
    print(f"PC LSH: {PC} PQ LSH: {PQ} F1 LSH: {F1StarLSH}")
    N=len(data)
    totalComparisons=(N*(N-1))/2
    print(f"total comparisons {totalComparisons} and fraction: {numberCandidates/totalComparisons}")
    return [PC, PQ, F1StarLSH]


def numberOfPairs(data):
    total = 0
    for i, word1 in enumerate(data):
        for j, word2 in enumerate(data):
            if word1[0] == word2[0]:
                if i < j:
                    total = total + 1
    return total

def allCombinations(data):
    result=[]
    for i, word1 in enumerate(data):
        for j, word2 in enumerate(data):
            if i < j:
                result.append((i,j))
    return result


if __name__ == "__main__":
    #sys.argv[1]
    main(10)
