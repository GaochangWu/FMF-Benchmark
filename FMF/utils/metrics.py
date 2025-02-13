"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np


class ClassificationMetric:
    # 本代码计算的指标仅适用于二分类
    def __init__(self, numClass=2):
        assert numClass == 2, 'numClass must be 2'
        self.numClass = numClass
        # confusionMatrix = [[TN, FP],
        #                    [FN, TP]]
        # 行为label，列为预测
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, clsPredict, clsLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (clsLabel >= 0) & (clsLabel < self.numClass)
        label = self.numClass * clsLabel[mask] + clsPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, clsPredict, clsLabel):
        assert clsPredict.shape == clsLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(clsPredict, clsLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def Accuracy(self):
        # 所有类别的分类准确率
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / (self.confusionMatrix.sum() + 1e-6)
        return acc

    def F1Score(self):
        # F1 score = 2 * Precision * Recall / (Precision + Recall)
        p = self.Precision()
        r = self.Recall()
        f1 = 2 * p * r / (p + r + 1e-5)
        return f1

    def Precision(self):
        # 精准率，预测的正样本中有多少是真实的正样本。值越大，性能越好
        # Precision =  TP / (TP + FP)
        p = self.confusionMatrix[1][1] / (self.confusionMatrix[1][1] + self.confusionMatrix[0][1] + 1e-6)
        return p

    def Recall(self):
        # 召回率，正样本被预测为正样本占总的正样本的比例。值越大，性能越好
        # Recall = TP / (TP + FN))
        r = self.confusionMatrix[1][1] / (self.confusionMatrix[1][1] + self.confusionMatrix[1][0] + 1e-6)
        return r

    def FalsePositiveRate(self):
        # 假阳率、误报率, 负样本被预测为正样本占总的负样本的比例。值越小, 性能越好
        # FPR = FP / (FP + TN)
        fpr = self.confusionMatrix[0][1] / (self.confusionMatrix[0][1] + self.confusionMatrix[0][0] + 1e-6)
        return fpr

    def FalseNegativeRate(self):
        # 假阴率、漏报率，正样本被预测为负样本占总的正样本的比例。值越小，性能越好
        # FNR = FN / (TP + FN)
        fnr = self.confusionMatrix[1][0] / (self.confusionMatrix[1][1] + self.confusionMatrix[1][0] + 1e-6)
        return fnr


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)
        self.samples = 0
        self.correct = 0

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / (self.confusionMatrix.sum() + 1e-5)
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + 1e-5)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / (union + 1e-5)  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def frameClassificationAccuracy(self):
        return self.correct / self.samples

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def updateSamplesCorrect(self, imgPredict, imgLabel):
        # 计算预测的label
        y_pred = np.sum(imgPredict, axis=(1, 2))
        y_pred[y_pred > 0] = 1

        # 计算正确的label
        y = np.sum(imgLabel, axis=(1, 2))
        y[y > 0] = 1

        self.correct += (y_pred == y).sum()
        self.samples += y.shape[0]

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        self.updateSamplesCorrect(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
        self.samples = 0
        self.correct = 0


class Container:
    def __init__(self, metricNames):
        self.metricNames = metricNames
        self.container = {mn: [] for mn in metricNames}

    def addEpoch(self, metricNums):
        assert len(metricNums) == len(self.metricNames)

        for k, num in zip(self.metricNames, metricNums):
            self.container[k].append(num)
