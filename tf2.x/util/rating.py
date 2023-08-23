import numpy as np
from util.config import ModelConf,OptionConf
import random
from collections import defaultdict
class Rating(object):
    'data access control'
    def __init__(self,config,trainingSet, testSet):
        self.config = config
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.drug = {} #map drug names to id
        self.disease = {} #map disease names to id
        self.id2drug = {}
        self.id2disease = {}
        self.drugMeans = {}
        self.diseaseMeans = {}
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict) #test set in the form of [drug][disease]=rating
        self.testSet_i = defaultdict(dict) #test set in the form of [disease][drug]=rating
        self.rScale = []
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]
        self.__generateSet()
        self.__computediseaseMean()
        self.__computedrugMean()
        self.__globalAverage()
        if self.evalSettings.contains('-cold'):
            self.__cold_start_test()


    def __generateSet(self):
        scale = set()
        if self.evalSettings.contains('-val'):
            random.shuffle(self.trainingData)
            separation = int(self.elemCount()*float(self.evalSettings['-val']))
            self.testData = self.trainingData[:separation]
            self.trainingData = self.trainingData[separation:]
        for i,entry in enumerate(self.trainingData):
            drugName,diseaseName,rating = entry
            if drugName not in self.drug:
                self.drug[drugName] = len(self.drug)
                self.id2drug[self.drug[drugName]] = drugName
            if diseaseName not in self.disease:
                self.disease[diseaseName] = len(self.disease)
                self.id2disease[self.disease[diseaseName]] = diseaseName
            self.trainSet_u[drugName][diseaseName] = rating
            self.trainSet_i[diseaseName][drugName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()

        for entry in self.testData:
            # drugName,diseaseName,rating = entry
            # if drugName not in self.drug:
            #     self.drug[drugName] = len(self.drug)
            #     self.id2drug[self.drug[drugName]] = drugName
            # if diseaseName not in self.disease:
            #     self.disease[diseaseName] = len(self.disease)
            #     self.id2disease[self.disease[diseaseName]] = diseaseName

            if self.evalSettings.contains('-predict'):
                self.testSet_u[entry]={}
            else:
                drugName, diseaseName, rating = entry
                self.testSet_u[drugName][diseaseName] = rating
                self.testSet_i[diseaseName][drugName] = rating

    def __cold_start_test(self):
        threshold = int(self.evalSettings['-cold'])
        removeddrug = {}
        for drug in self.testSet_u:
            if drug in self.trainSet_u and len(self.trainSet_u[drug])>threshold:
                removeddrug[drug]=1
        for drug in removeddrug:
            del self.testSet_u[drug]
        testData = []
        for disease in self.testData:
            if disease[0] not in removeddrug:
                testData.append(disease)
        self.testData = testData

    def __globalAverage(self):
        total = sum(self.drugMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.drugMeans)

    def __computedrugMean(self):
        for u in self.drug:
            self.drugMeans[u] = sum(self.trainSet_u[u].values())/len(self.trainSet_u[u])

    def __computediseaseMean(self):
        for c in self.disease:
            self.diseaseMeans[c] = sum(self.trainSet_i[c].values())/len(self.trainSet_i[c])

    def getdrugId(self,u):
        if u in self.drug:
            return self.drug[u]

    def getdiseaseId(self,i):
        if i in self.disease:
            return self.disease[i]

    def trainingSize(self):
        return (len(self.drug),len(self.disease),len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether drug u rated disease i'
        if u in self.drug and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containsdrug(self,u):
        'whether drug is in training set'
        if u in self.drug:
            return True
        else:
            return False

    def containsdisease(self,i):
        'whether disease is in training set'
        if i in self.disease:
            return True
        else:
            return False

    def drugRated(self,u):
        return list(self.trainSet_u[u].keys()),list(self.trainSet_u[u].values())

    def diseaseRated(self,i):
        return list(self.trainSet_i[i].keys()),list(self.trainSet_i[i].values())

    def row(self,u):
        k,v = self.drugRated(u)
        vec = np.zeros(len(self.disease))
        #print vec
        for pair in zip(k,v):
            iid = self.disease[pair[0]]
            vec[iid]=pair[1]
        return vec

    def col(self,i):
        k,v = self.diseaseRated(i)
        vec = np.zeros(len(self.drug))
        #print vec
        for pair in zip(k,v):
            uid = self.drug[pair[0]]
            vec[uid]=pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.drug),len(self.disease)))
        for u in self.drug:
            k, v = self.drugRated(u)
            vec = np.zeros(len(self.disease))
            # print vec
            for pair in zip(k, v):
                iid = self.disease[pair[0]]
                vec[iid] = pair[1]
            m[self.drug[u]]=vec
        return m

    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
