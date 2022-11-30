"""定义basic类"""
import pandas as pd
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import xlwt


class Basic:
    """实现基础变量与函数"""
    def __init__(self):
        """构造函数"""
        # 读取数据
        df = pd.read_excel("data.xlsx", usecols=np.arange(33), names=None)
        # 定义数据的正负样本数
        self.positive_num = 212
        self.negative_num = 808
        # 定义特征向量X
        self.X = df.values.tolist()
        # 定义标签Y
        df = pd.read_excel("data.xlsx", usecols=[33, 34], names=None)
        labels = df.values.tolist()
        self.Y = []
        # 当病人死亡或出现MACCE时，记为1；否则记为0
        for label in labels:
            if sum(label) == 0:
                self.Y.append(0)
            else:
                self.Y.append(1)
        # 定义可供使用的算法
        self.methods = ['LogisticRegression', 'DecisionTree', 'SVM', 'RandomForest', 'Xgboost', 'MLP', 'Adaboost']

    def predict(self, method):
        """使用留一法进行预测"""
        # 定义预测结果
        pred = []
        pred_proba = []
        # 留一法
        i = 0
        while i < 1020:
            train_x = copy.deepcopy(self.X)
            train_y = copy.deepcopy(self.Y)
            test_x = copy.deepcopy(self.X[i])
            test_y = self.Y[i]
            del train_x[i]
            del train_y[i]
            # 打乱训练集
            randnum = random.randint(0, 100)
            random.seed(randnum)
            random.shuffle(train_x)
            random.seed(randnum)
            random.shuffle(train_y)
            # 训练
            if method == 'LogisticRegression':
                clf = LogisticRegression(penalty='l2', C=1, class_weight='balanced', max_iter=5000)
                clf.fit(train_x, train_y)
                label = clf.predict([test_x])
                proba = clf.predict_proba([test_x])
            if method == 'DecisionTree':
                clf = DecisionTreeClassifier(criterion='gini', max_depth=3, class_weight='balanced')
                clf.fit(train_x, train_y)
                label = clf.predict([test_x])
                proba = clf.predict_proba([test_x])
            if method == 'SVM':
                clf = svm.SVC(C=0.8, kernel='rbf', class_weight={0: 1, 1: 4}, probability=True)
                clf.fit(train_x, train_y)
                label = clf.predict([test_x])
                proba = clf.predict_proba([test_x])
            if method == 'RandomForest':
                clf = RandomForestClassifier(n_estimators=300, max_features='sqrt', max_depth=5,
                                             class_weight={0: 1, 1: 4})
                clf.fit(train_x, train_y)
                label = clf.predict([test_x])
                proba = clf.predict_proba([test_x])
            if method == 'Xgboost':
                clf = XGBClassifier(
                    # 指定叶节点进行分支所需的损失减少的最小值, 设置的值越大，模型就越保守
                    gamma=1,
                    # 树的最大深度, [3, 5, 6, 7, 9, 12, 15, 17, 25]
                    max_depth=5,
                    # 孩子节点中最小的样本权重和, [1, 3, 5, 7]
                    min_child_weight=3,
                    # 采样出 subsample * n_samples 个样本用于训练弱学习器(不放回抽样)
                    # 选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低, [0.6, 0.7, 0.8, 0.9, 1]
                    subsample=0.8,
                    # 构建弱学习器时，对特征随机采样的比例, [0.6, 0.7, 0.8, 0.9, 1]
                    colsample_bytree=0.8,
                    # 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
                    scale_pos_weight='balanced',
                    # 学习率
                    learning_rate=0.01,
                    # 使用的弱分类器个数
                    n_estimators=20,
                    # 用于指定学习任务及相应的学习目标
                    objective='multi:softmax',
                    # 弱学习器的类型('gbtree', 'gblinear')
                    booster='gbtree',
                    # 用于设置多分类问题的类别个数
                    num_class=2,
                    # L1正则化权重项，增加此值将使模型更加保守, [0, 0.01~0.1, 1]
                    reg_alpha=0.01,
                    # L2正则化权重项，增加此值将使模型更加保守, [0, 0.1, 0.5, 1]
                    reg_lambda=0.01,
                    # 指定随机数种子
                    seed=0,
                    max_delta_step=0,
                )
                # eval_metric用于指定评估指标
                clf.fit(train_x, train_y, eval_set=[(train_x, train_y), ([test_x], [test_y])], eval_metric='mlogloss')
                label = clf.predict([test_x])
                proba = clf.predict_proba([test_x])
            if method == 'MLP':
                clf = MLPClassifier(hidden_layer_sizes=(50, 10), max_iter=500, tol=1e-4)
                clf.fit(train_x, train_y)
                label = clf.predict([test_x])
                proba = clf.predict_proba([test_x])
            if method == 'Adaboost':
                clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5,
                                                                class_weight={0: 1, 1: 15}),
                                         n_estimators=50, learning_rate=1)
                clf.fit(train_x, train_y)
                label = clf.predict([test_x])
                proba = clf.predict_proba([test_x])
            pred.append(label[0])
            pred_proba.append(proba[0][1])
            del train_x
            del train_y
            del test_x
            del test_y
            del clf
            i += 1
        return pred, pred_proba

    def print_results(self, pred):
        """输出混淆矩阵, 正确率和错误编号"""
        i = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        # 定义错误编号
        error_index = []
        while i < 1020:
            if self.Y[i] == 1:
                if pred[i] == 1:
                    TP += 1
                else:
                    FN += 1
                    error_index.append(i + 1)
            else:
                if pred[i] == 1:
                    FP += 1
                    error_index.append(i + 1)
                else:
                    TN += 1
            i += 1
        print("混淆矩阵:")
        print(str(TP) + "    " + str(FN))
        print(str(FP) + "    " + str(TN))
        print("正确率:")
        print((TP + TN) / (TP + FN + FP + TN))
        return error_index

    def analyse_error_index(self, list_1, list_2):
        """对不同分类器得到的错误编号进行分析"""
        set_1_2 = set(list_1) & set(list_2)
        list_1_2 = list(set_1_2)
        return list_1_2

    def plot_roc(self, pred_proba, method):
        """绘制ROC曲线,并计算AUC"""
        fpr, tpr, thresholds = roc_curve(self.Y, pred_proba, pos_label=1)
        AUC = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % AUC)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(method + "  ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

    def write_error_index(self, error_index):
        """将错误编号写入excel"""
        f = xlwt.Workbook('encoding = utf-8')
        sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
        for i in range(len(error_index)):
            # 写入数据参数对应 行, 列, 值
            sheet1.write(i, 0, error_index[i])
        f.save('error.xlsx')
