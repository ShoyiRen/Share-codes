#这份代码是在我刚刚学习Python这门语言的时候写的，是有关于一种基本的机器学习算法———贝叶斯网络的。
#尽管这份代码的有些实现细节不是特别的简洁，但依旧有两个原因让我选择分享这份代码， 
#一方面是认为这份代码能够体现出我较强的学习能力(例如：学习新语言或者新算法)，
#另一方面是觉得这份代码能够比较一定程度上体现出我的编程习惯
#由于Python是依赖缩进进行编译的，所以整份代码的结构显得十分有层次
#另一方面，代码中的变量名和函数名尽管都很长，但其名字都明显符合这个函数或者变量所拥有的功能或者身份
#这也是为何代码中不存在太多注释的原因，因为我觉得通过函数和变量名就能够很好的理解整个函数的作用

#Zeping Ren


import scipy.io.arff as arff
import sys
import math
import random

def loadingdata(path):
    loaded_arff = arff.loadarff(open(path, 'rb'))
    (training_data, metadata) = loaded_arff
    return training_data, metadata

#implementation of Naive Bayes
class NaiveBayes(object):
    def DataClassDivision(self, TrainingData, TrainingDataType):
        FirstClassData = []
        SecondClassData = []
        for EachTrainingData in TrainingData:
            if EachTrainingData['class'] == TrainingDataType['class'][1][0]:
                FirstClassData.append(EachTrainingData)
            else:
                SecondClassData.append(EachTrainingData)
        return(FirstClassData, SecondClassData)

    def CondPossibilityEstimation(self, FirstClassData, SecondClassData, NaiveBayesNetwork, TrainingDataType):
        for EachFeatureName in TrainingDataType:
            if EachFeatureName == 'class':
                continue
            NumofFeatureValues = len(TrainingDataType[EachFeatureName][1])
            
            FirstDataCounts = [1 for x in range(NumofFeatureValues)]
            for EachFirstClassData in FirstClassData:
                FeatureID = TrainingDataType[EachFeatureName][1].index(EachFirstClassData[EachFeatureName])
                FirstDataCounts[FeatureID] += 1
            for i in range(len(FirstDataCounts)):
                FirstDataCounts[i] = float(FirstDataCounts[i]) / float (NumofFeatureValues + len(FirstClassData))

            SecondDataCounts = [1 for x in range(NumofFeatureValues)]
            for EachSecondClassData in SecondClassData:
                FeatureID = TrainingDataType[EachFeatureName][1].index(EachSecondClassData[EachFeatureName])
                SecondDataCounts[FeatureID] += 1
            for i in range(len(SecondDataCounts)):
                SecondDataCounts[i] = float(SecondDataCounts[i]) / float(NumofFeatureValues + len(SecondClassData))
            
            
            NaiveBayesNetwork[EachFeatureName] = [FirstDataCounts, SecondDataCounts]
        return

    def NaiveBayesTraining(self, TrainingData, TrainingDataType):
        (FirstClassData, SecondClassData) = self.DataClassDivision(TrainingData, TrainingDataType)
        NaiveBayesNetwork = {}
        NaiveBayesNetwork['class'] = [float(len(FirstClassData) + 1)/float(len(TrainingData) + 2), \
                                     float(len(SecondClassData) + 1)/float(len(TrainingData) + 2)]
        self.CondPossibilityEstimation(FirstClassData, SecondClassData, NaiveBayesNetwork, TrainingDataType)
        return NaiveBayesNetwork

    def NaiveBayesTest(self, TestData, TestDataType, NaiveBayesNetwork):
        NaiveBayesResult = []
        for EachTestData in TestData:
            FirstClassPossibility = NaiveBayesNetwork['class'][0]
            SecondClassPossibility = NaiveBayesNetwork['class'][1]
            for EachFeatureName in TestDataType:
                if EachFeatureName == 'class':
                    continue
                FeatureID = TestDataType[EachFeatureName][1].index(EachTestData[EachFeatureName])
                FirstClassPossibility *= NaiveBayesNetwork[EachFeatureName][0][FeatureID]
                SecondClassPossibility *= NaiveBayesNetwork[EachFeatureName][1][FeatureID]
            if(FirstClassPossibility > SecondClassPossibility):
                NaiveBayesResult.append([TestDataType['class'][1][0] , EachTestData['class'], \
                                        FirstClassPossibility / (FirstClassPossibility + SecondClassPossibility)])
            else:
                NaiveBayesResult.append([TestDataType['class'][1][1] , EachTestData['class'], \
                                        SecondClassPossibility / (FirstClassPossibility + SecondClassPossibility)])
        return NaiveBayesResult

    def NaiveBayesPrint(self, NaiveBayesResult, DataType):
        for EachFeatureName in DataType:
            if EachFeatureName == 'class':
                continue
        CorrectNum = 0
        for EachNaiveBayesResult in  NaiveBayesResult:
            if EachNaiveBayesResult[0][0] == '\'':
                EachNaiveBayesResult[0] = EachNaiveBayesResult[0][1:-1]
            if EachNaiveBayesResult[1][0] == '\'':
                EachNaiveBayesResult[1] = EachNaiveBayesResult[1][1:-1]
            EachNaiveBayesResult[2] = round(EachNaiveBayesResult[2],12)
            if EachNaiveBayesResult[0] == EachNaiveBayesResult[1]:
                CorrectNum += 1
        return CorrectNum

#Implementation of TAN algorithm
class TANBayes(object):
    def DataClassDivision(self, TrainingData, TrainingDataType):
        FirstClassData = []
        SecondClassData = []
        for EachTrainingData in TrainingData:
            if EachTrainingData['class'] == TrainingDataType['class'][1][0]:
                FirstClassData.append(EachTrainingData)
            else:
                SecondClassData.append(EachTrainingData)
        return(FirstClassData, SecondClassData)

    def MutualInformationGet(self, FirstClassData, SecondClassData, NumofDatas, FeatureNameI, FeatureNameJ, CondPossibilityPerFearture, TrainingDataType):
        NumofFeatureValuesI  = len(TrainingDataType[FeatureNameI][1])
        NumofFeatureValuesJ  = len(TrainingDataType[FeatureNameJ][1])
        NumofFeatureValues = NumofFeatureValuesI * NumofFeatureValuesJ
        
        FirstDataCounts = [1 for x in range(NumofFeatureValues)]
        TotalFirstCounts = [1 for x in range(NumofFeatureValues)]
        for EachFirstClassData in FirstClassData:
            FeatureIDI = TrainingDataType[FeatureNameI][1].index(EachFirstClassData[FeatureNameI])
            FeatureIDJ = TrainingDataType[FeatureNameJ][1].index(EachFirstClassData[FeatureNameJ])
            FirstDataCounts[FeatureIDI * NumofFeatureValuesJ + FeatureIDJ] += 1
            TotalFirstCounts[FeatureIDI * NumofFeatureValuesJ + FeatureIDJ] += 1
        for i in range(len(FirstDataCounts)):
            FirstDataCounts[i] = float(FirstDataCounts[i]) / float (NumofFeatureValues + len(FirstClassData))
        
        SecondDataCounts = [1 for x in range(NumofFeatureValues)]
        TotalSecondCounts = [1 for x in range(NumofFeatureValues)]
        for EachSecondClassData in SecondClassData:
            FeatureIDI = TrainingDataType[FeatureNameI][1].index(EachSecondClassData[FeatureNameI])
            FeatureIDJ = TrainingDataType[FeatureNameJ][1].index(EachSecondClassData[FeatureNameJ])
            SecondDataCounts[FeatureIDI * NumofFeatureValuesJ + FeatureIDJ] += 1
            TotalSecondCounts[FeatureIDI * NumofFeatureValuesJ + FeatureIDJ] += 1
        for i in range(len(SecondDataCounts)):
            SecondDataCounts[i] = float(SecondDataCounts[i]) / float(NumofFeatureValues + len(SecondClassData))
        

        TotalDataCounts = []
        for i in range(NumofFeatureValues):
            TotalFirst = float(TotalFirstCounts[i]) / float(len(FirstClassData) + len(SecondClassData) + NumofFeatureValues * 2)
            TotalSecond = float(TotalSecondCounts[i]) / float(len(FirstClassData) + len(SecondClassData) + NumofFeatureValues * 2)
            TotalDataCounts.append([TotalFirst, TotalSecond])
        
        EachMutualInformation = 0.0
        for xi in range(NumofFeatureValuesI):
            for xj in range(NumofFeatureValuesJ):
                idx = xi * NumofFeatureValuesJ + xj
                EachMutualInformation += TotalDataCounts[idx][0]\
                                                    * (math.log(FirstDataCounts[idx]) - math.log(CondPossibilityPerFearture[FeatureNameI][0][xi]) \
                                                    - math.log(CondPossibilityPerFearture[FeatureNameJ][0][xj]))
                EachMutualInformation += TotalDataCounts[idx][1]\
                                                    * (math.log(SecondDataCounts[idx]) - math.log(CondPossibilityPerFearture[FeatureNameI][1][xi]) \
                                                    - math.log(CondPossibilityPerFearture[FeatureNameJ][1][xj]))
        return EachMutualInformation / math.log(2)

    def GenerateMST(self, MutualInformation, AdjEdges, NumofFeatureNames):
        ValidNode = [1 for x in range(NumofFeatureNames)]
        ValidNode[0] = 0
        MI = MutualInformation[0]
    
        for j in range(len(AdjEdges)-1):
            maxMI = -0.5
            maxMIfromidx = -1
            maxMItoidx = -1
            for i in range(len(AdjEdges)):
                if ValidNode[i] == 0:
                    continue
                if MI[i] > maxMI:
                    maxMI = MI[i]
                    maxMIfromidx = AdjEdges[i]
                    maxMItoidx = i
                elif MI[i] == maxMI:
                    if maxMIfromidx > AdjEdges[i]:
                        maxMIfromidx = AdjEdges[i]
                        maxMItoidx = i
                            
            ValidNode[maxMItoidx] = 0
            for i in range(len(AdjEdges)):
                if ValidNode[i] == 0:
                    continue
                if MI[i] < MutualInformation[maxMItoidx][i]:
                    MI[i] = MutualInformation[maxMItoidx][i]
                    AdjEdges[i] = maxMItoidx
        return MI

    def TANCondPossibilityEstimation(self, FirstClassData, SecondClassData, TANNetwork, AdjEdges, CondPossibilityPerFearture, TrainingDataType, FeatureNames):
        TANNetwork['class'] = CondPossibilityPerFearture['class']
        TANNetwork[FeatureNames[0]] = CondPossibilityPerFearture[FeatureNames[0]]
        for i in range(1, len(FeatureNames)):
            FromFeatureName = FeatureNames[AdjEdges[i]]
            ToFeatureName = FeatureNames[i]
            NumofFromFeatureValues = len(TrainingDataType[FromFeatureName][1])
            NumofToFeatureValues = len(TrainingDataType[ToFeatureName][1])
            FirstToFeaturePossibility = []
            SecondToFeaturePossibility = []

            for j in range(NumofFromFeatureValues):
                NumofFirstClassAndFromValue = NumofToFeatureValues
                ToFeaturePossibilityEachCondition = [1 for x in range(NumofToFeatureValues)]
                for EachFirstClassData in FirstClassData:
                    if EachFirstClassData[FromFeatureName] == TrainingDataType[FromFeatureName][1][j]:
                        FeatureID = TrainingDataType[ToFeatureName][1].index(EachFirstClassData[ToFeatureName])
                        ToFeaturePossibilityEachCondition[FeatureID] += 1
                        NumofFirstClassAndFromValue += 1
                for i in range(len(ToFeaturePossibilityEachCondition)):
                    ToFeaturePossibilityEachCondition[i] = float(ToFeaturePossibilityEachCondition[i]) / float(NumofFirstClassAndFromValue)
                FirstToFeaturePossibility.append(ToFeaturePossibilityEachCondition)

            for j in range(NumofFromFeatureValues):
                NumofSecondClassAndFromValue = NumofToFeatureValues
                ToFeaturePossibilityEachCondition = [1 for x in range(NumofToFeatureValues)]
                for EachSecondClassData in SecondClassData:
                    if EachSecondClassData[FromFeatureName] == TrainingDataType[FromFeatureName][1][j]:
                        FeatureID = TrainingDataType[ToFeatureName][1].index(EachSecondClassData[ToFeatureName])
                        ToFeaturePossibilityEachCondition[FeatureID] += 1
                        NumofSecondClassAndFromValue += 1
                for i in range(len(ToFeaturePossibilityEachCondition)):
                    ToFeaturePossibilityEachCondition[i] = float(ToFeaturePossibilityEachCondition[i]) / float(NumofSecondClassAndFromValue)
                SecondToFeaturePossibility.append(ToFeaturePossibilityEachCondition)
        
            TANNetwork[ToFeatureName] = [FirstToFeaturePossibility, SecondToFeaturePossibility]
        return

    def TANTraining(self, TrainingData, TrainingDataType):
        NaiveBayesTra = NaiveBayes()
        (FirstClassData, SecondClassData) = self.DataClassDivision(TrainingData, TrainingDataType)
        
        FeatureNames = []
        for EachFeatureName in TrainingDataType:
            if EachFeatureName == 'class':
                continue
            FeatureNames.append(EachFeatureName)
        NumofFeatureNames = len(FeatureNames)

        CondPossibilityPerFearture = {}
        CondPossibilityPerFearture['class'] = [float(len(FirstClassData) + 1)/float(len(TrainingData) + 2), \
                                               float(len(SecondClassData) + 1)/float(len(TrainingData) + 2)]
        NaiveBayesTra.CondPossibilityEstimation(FirstClassData, SecondClassData, CondPossibilityPerFearture, TrainingDataType)
        MutualInformation = []
        for i in range(NumofFeatureNames):
                MutualInformation.append([-1 for x in range(NumofFeatureNames)])

        for i in range(NumofFeatureNames):
            for j in range(i+1, NumofFeatureNames):
                MutualInformation[i][j] = self.MutualInformationGet(FirstClassData, SecondClassData, len(TrainingData), \
                                                               FeatureNames[i], FeatureNames[j], CondPossibilityPerFearture, TrainingDataType)
                MutualInformation[j][i] = MutualInformation[i][j]
 
        AdjEdges = [0 for x in range(NumofFeatureNames)]
        AdjEdges[0] = 'None'
        MI = self.GenerateMST(MutualInformation, AdjEdges, len(FeatureNames))


        TANNetwork = {}
        self.TANCondPossibilityEstimation(FirstClassData, SecondClassData, TANNetwork, AdjEdges, CondPossibilityPerFearture, TrainingDataType, FeatureNames)
        return (TANNetwork, AdjEdges, FeatureNames)

    def TANTest(self, TestData, TestDataType, TANNetwork, AdjEdges, FeatureNames):
        TANResult = []
        for EachTestData in TestData:
            FirstClassPossibility = TANNetwork['class'][0]
            SecondClassPossibility = TANNetwork['class'][1]
            for i in range(len(FeatureNames)):
                ChildFeatureName = FeatureNames[i]
                if ChildFeatureName  == 'class':
                    continue
                ChildFeatureID = TestDataType[ChildFeatureName][1].index(EachTestData[ChildFeatureName])
                
                if AdjEdges[i] == 'None':
                    FirstClassPossibility *= TANNetwork[ChildFeatureName][0][ChildFeatureID]
                    SecondClassPossibility *= TANNetwork[ChildFeatureName][1][ChildFeatureID]
                else:
                    ParentFeatureName = FeatureNames[AdjEdges[i]]
                    ParentFeatureID = TestDataType[ParentFeatureName][1].index(EachTestData[ParentFeatureName])
                    FirstClassPossibility *= TANNetwork[ChildFeatureName][0][ParentFeatureID][ChildFeatureID]
                    SecondClassPossibility *= TANNetwork[ChildFeatureName][1][ParentFeatureID][ChildFeatureID]

            if(FirstClassPossibility > SecondClassPossibility):
                TANResult.append([TestDataType['class'][1][0] , EachTestData['class'], \
                                  FirstClassPossibility / (FirstClassPossibility + SecondClassPossibility)])
            else:
                TANResult.append([TestDataType['class'][1][1] , EachTestData['class'], \
                                  SecondClassPossibility / (FirstClassPossibility + SecondClassPossibility)])
        return TANResult

    def TANPrint(self, TANResult, AdjEdges, FeatureNames):
        for i in range(len(FeatureNames)):
            ChildFeatureName = FeatureNames[i]
            if ChildFeatureName == 'class':
                continue
            if AdjEdges[i] == 'None':
                pass
                print ChildFeatureName + ' class'
            else:
                ParentFeatureName = FeatureNames[AdjEdges[i]]
            
            CorrectNum = 0
        for EachTANResult in  TANResult:
            if EachTANResult[0][0] == '\'':
                EachTANResult[0] = EachTANResult[0][1:-1]
            if EachTANResult[1][0] == '\'':
                EachTANResult[1] = EachTANResult[1][1:-1]
            EachTANResult[2] = round(EachTANResult[2],12)
            if EachTANResult[0] == EachTANResult[1]:
                CorrectNum += 1
        return CorrectNum

#Cross validation on both Naive Bayes and TAN
class CrossValidation(object):
    def DataIdxPartition(self, TrainingData, TrainingDataType):
        FirstClassIdx = []
        SecondClassIdx = []
        for i in range(len(TrainingData)):
            if TrainingData[i]['class'] == TrainingDataType['class'][1][0]:
                FirstClassIdx.append(i)
            else:
                SecondClassIdx.append(i)
        return FirstClassIdx, SecondClassIdx
    
    def DataPartition(self, TrainingData, FirstClassIdx, SecondClassIdx):
        NumofFirstClassData = len(FirstClassIdx) / 10.0
        NumofSecondClassData = len(SecondClassIdx) / 10.0
        StepSizeFirst = int(math.ceil(NumofFirstClassData))
        StepSizeSecond = int(math.ceil(NumofSecondClassData))
        random.shuffle(FirstClassIdx)
        random.shuffle(SecondClassIdx)
        TData = []
        VData = []
        for i in range(10):
            TempTData = []
            TempVData = []
            for j in range(len(FirstClassIdx)):
                if j%10 == i:
                    TempVData.append(TrainingData[FirstClassIdx[j]])
                else:
                    TempTData.append(TrainingData[FirstClassIdx[j]])
            for j in range(len(SecondClassIdx)):
                if j%10 == i:
                    TempVData.append(TrainingData[SecondClassIdx[j]])
                else:
                    TempTData.append(TrainingData[SecondClassIdx[j]])

            TData.append(TempTData)
            VData.append(TempVData)
        return (TData, VData)

    
    def CrossValidationUtility(self, TrainingData, TrainingDataType):
        CVNaive = NaiveBayes()
        CVTAN = TANBayes()
        
        (FirstClassIdx, SecondClassIdx) = self.DataIdxPartition(TrainingData, TrainingDataType)
        (TData, VData) = self.DataPartition(TrainingData, FirstClassIdx, SecondClassIdx)
        for i in range(10):
            NaiveBayesNetwork = CVNaive.NaiveBayesTraining(TData[i], TrainingDataType)
            NaiveBayesResult = CVNaive.NaiveBayesTest(VData[i], TrainingDataType, NaiveBayesNetwork)
            NaiveCorrect = CVNaive.NaiveBayesPrint(NaiveBayesResult, TrainingDataType)
            
            (TANNetwork, AdjEdges, FeatureNames) = CVTAN.TANTraining(TData[i], TrainingDataType)
            TANResult = CVTAN.TANTest(VData[i], TrainingDataType, TANNetwork, AdjEdges, FeatureNames)
            TANCorrect = CVTAN.TANPrint(TANResult, AdjEdges, FeatureNames)
            
            NaiveCorrect = float(NaiveCorrect) / len(VData[i])
            TANCorrect = float(TANCorrect) / len(VData[i])
            print NaiveCorrect, TANCorrect

# main function
def bayes():  
    #defalut config: input filename and algoithm
    TrainingFileName = 'lymph_train.arff'
    TestFileName = 'lymph_test.arff'
    NT = 'cv'

    #load data
    TrainingData, TrainingDataType = loadingdata(TrainingFileName)
    TestData, TestDataType = loadingdata(TestFileName)
    
    if NT == 'n':
        Naive = NaiveBayes()
        NaiveBayesNetwork = Naive.NaiveBayesTraining(TrainingData, TrainingDataType)
        NaiveBayesResult = Naive.NaiveBayesTest(TestData, TestDataType, NaiveBayesNetwork)#
    if NT == 't':
        TAN = TANBayes()
        (TANNetwork, AdjEdges, FeatureNames) = TAN.TANTraining(TrainingData, TrainingDataType)
        TANResult = TAN.TANTest(TestData, TestDataType, TANNetwork, AdjEdges, FeatureNames)
    if NT == 'cv':
        TrainingData, TrainingDataType = loadingdata('chess-KingRookVKingPawn.arff')
        CV = CrossValidation()
        CV.CrossValidationUtility(TrainingData, TrainingDataType)
    return

if __name__=="__main__":
    bayes()

