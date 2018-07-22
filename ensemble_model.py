import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 内嵌画图，可省略plt.show()
# % matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from collections import Counter

# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')

# 2.1 load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
IDtest = test["PassengerId"]

# Outlier detection 异常值检测
def detect_outliers(df, n, features):
    outlier_indices = []

    # 迭代特征
    for col in features:
        # 第一四分位数（25%）
        Q1 = np.percentile(df[col], 25, axis=0, interpolation='lower')
        # 第三四分位数（75%）
        Q3 = np.percentile(df[col], 75, axis=0, interpolation='lower')
        # 四分位间距(IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5*IQR

        # 确定特征列的异常值索引列表
        outlier_list_col = df[(df[col] < (Q1-outlier_step)) | (df[col] > (Q3+outlier_step))].index

        # 将找到的col异常值索引附加到异常值索引列表中
        outlier_indices.extend(outlier_list_col)

        # 选择包含n个以上异常值的观测值
        outlier_indices = Counter(outlier_indices)
        multiple_outlier = list(k for k, v in outlier_indices.items() if v > n)

        return multiple_outlier

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
# 打印异常值行
print(train.loc[Outliers_to_drop])

# 2.3 joining train and test set
train_len = len(train)
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# 2.4 check for null and missing values
dataset = dataset.fillna(np.nan)
# print(dataset.isnull().sum())


# 3. feature analysis
# 3.1 numerical values

# 数值特征（SibSp Parch年龄和票价值）和Survived间的相关矩阵
# g = sns.heatmap(train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.show()

# Fare特征值

dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
# print(dataset["Fare"].isnull().sum())
# print(dataset["Fare"].min())

# 给Fare增加Fare_cat特征
dataset["Fare_cat"] = 0
dataset.loc[dataset["Fare"]<=7.91, "Fare_cat"] = 0 
dataset.loc[(dataset["Fare"]>7.91) & (dataset["Fare"]<=14.454), "Fare_cat"] = 1
dataset.loc[(dataset["Fare"]>14.454) & (dataset["Fare"]<=31), "Fare_cat"] = 2
dataset.loc[(dataset["Fare"]>31) & (dataset["Fare"]<=513), "Fare_cat"] = 3


# 3.2 类目特征
# Sex
# g = sns.barplot(x="Sex", y="Survived", data=train)
# g.set_ylabel("Survived Probability")
# plt.show()

dataset["Embarked"] = dataset["Embarked"].fillna("S")
# print(dataset["Embarked"])
dataset["Embarked"].replace(['S', 'C', 'Q'], [0,1,2], inplace=True)
dataset["Sex"] = dataset["Sex"].map({"male":0, "female":1})

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][ ((dataset["SibSp"] == dataset.iloc[i]["SibSp"]) & (dataset["Parch"] == dataset.iloc[i]["Parch"]) & (dataset["Pclass"] == dataset.iloc[i]["Pclass"])) ].median()
    if not np.isnan(age_pred):
        dataset["Age"].iloc[i] = age_pred
    else:
        dataset["Age"].iloc[i] = age_med

# 给Age添加新特征Age_band
dataset["Age_band"] = 0
dataset.loc[dataset["Age"]<=16, "Age_band"] = 0
dataset.loc[(dataset["Age"]>16) & (dataset["Age"]<=32), "Age_band"] = 1
dataset.loc[(dataset["Age"]>32) & (dataset["Age"]<=48), "Age_band"] = 2
dataset.loc[(dataset["Age"]>48) & (dataset["Age"]<=64), "Age_band"] = 3
dataset.loc[dataset["Age"]>64, "Age_band"] = 4
# dataset.head()

# print(dataset["Age"].isnull().sum())

# 5 特征工程
# 5.1 get title from name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
# print(dataset["Title"].head())

# 将人的称号分类
# 首先将Lady,the Countess等比较少见的称呼替换为Rare
dataset["Title"] = dataset["Title"].replace(["Lady", "the Countess", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
# 然后将Title映射为数值类型
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)

# g = sns.countplot(dataset["Title"])
# g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])
# plt.show()

dataset.drop(labels=["Name"], axis=1, inplace=True)

# 创建一个family size特征值
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

# g = sns.factorplot(x="Fsize", y="Survived", data=dataset)
# g.set_ylabels("Survived Probability")
# plt.show()

# 为家庭大小创建四种类别
dataset["Single"] = dataset["Fsize"].map(lambda s: 1 if s==1 else 0)
dataset["SmallF"] = dataset["Fsize"].map(lambda s: 1 if s==2 else 0)
dataset["MedF"] = dataset["Fsize"].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset["LargeF"] = dataset["Fsize"].map(lambda s: 1 if s>=5 else 0)

# 特征因子化，特征有几种取值就将其分为几列
dataset = pd.get_dummies(dataset, columns=["Title"])
dataset = pd.get_dummies(dataset, columns=["Embarked"])


# 处理Cabin缺失值,且每个Cabin全部用第一个字符简单表示
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset["Cabin"]])

# g = sns.factorplot(y="Survived", x="Cabin", data=dataset, kind="bar", order=['A','B','C','D','E','F','G','T','X'])
# g.set_ylabels("Survival Probability")
# plt.show()

dataset = pd.get_dummies(dataset, columns=["Cabin"])
# print(dataset.head())

# 创建Ticket前缀特征,用一代替Ticket
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".", "").replace("/", "").strip().split(" ")[0])
    else:
        Ticket.append("X")

dataset["Ticket"] = Ticket
# g = sns.factorplot(y="Survived", x="Ticket", data=dataset, kind="bar")
# g.set_ylabels("Survival Probability")
# plt.show()
# print(dataset["Ticket"].unique())

dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")

# 特征化Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")

# 删除没用的特征
dataset.drop(["PassengerId", "SibSp", "Parch", "Age", "Fare"], axis=1, inplace=True)
# print(dataset.head())

# 6 model ensemble
# 将训练集和测试集分开
train2 = dataset[:train_len]
train2["Survived"] = train2["Survived"].astype(int)
# 将训练集分为两部分
# split_train, split_test = train_test_split(train2, test_size=0.3, random_state=0, stratify=train2["Survived"])
# split_train_X = split_train[split_train.columns[1:]]
# split_train_Y = split_train[split_train.columns[:1]] # 第一列是Survived吗？
# split_test_X = split_test[split_test.columns[1:]]
# split_test_Y = split_test[split_test.columns[:1]]
Y = train2["Survived"]
train2.drop(labels=["Survived"], axis=1, inplace=True)
X = train2

# Radial Support Vector Machines(rbf-SVM) 径向支持向量机（rbf-SVM）
# model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
# model.fit(split_train_X, split_train_Y)
# prediction1 = model.predict(split_test_X)
# print("Accuracy for rbf SVM is ", metrics.accuracy_score(prediction1, split_test_Y))

# Linear Support Vector Machine(linear-SVM)
# model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
# model.fit(split_train_X, split_train_Y)
# prediction2 = model.predict(split_test_X)
# print("Accuracy for linear SVM is ", metrics.accuracy_score(prediction2, split_test_Y))

# Logistic Regression
# model = LogisticRegression(penalty='l1')
# model.fit(split_train_X, split_train_Y)
# prediction3 = model.predict(split_test_X)
# print("Accuracy of LR is ", metrics.accuracy_score(prediction3, split_test_Y))

# Decision Tree
# model = DecisionTreeClassifier()
# model.fit(split_train_X, split_train_Y)
# prediction4 = model.predict(split_test_X)
# print("Accuracy of Decision Tree is ", metrics.accuracy_score(prediction4, split_test_Y))

# K-Nearest Neighbours(KNN)
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(split_train_X, split_train_Y)
# prediction5 = model.predict(split_test_X)
# print("Accuracy of the KNN is ", metrics.accuracy_score(prediction5, split_test_Y))

# KNN模型中n_neighbors取值对其精度影响较大,画图看下n_neighbors与精度的关系
# 画图得知n_neighbors=3时精确度最大
# a_index = list(range(1,11))
# a = pd.Series()
# x = [0,1,2,3,4,5,6,7,8,9,10]
# for i in list(range(1,11)):
#     model = KNeighborsClassifier(n_neighbors=i)
#     model.fit(split_train_X, split_train_Y)
#     prediction = model.predict(split_test_X)
#     a = a.append(pd.Series(metrics.accuracy_score(prediction, split_test_Y)))
# plt.plot(a_index, a)
# plt.xticks(x)
# fig = plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()
# print("Accuracy for different values of n are :", a.values, "the max value is ", a.values.max())

# Gaussian Naive Bayes 高斯朴素贝叶斯
# model = GaussianNB()
# model.fit(split_train_X, split_train_Y)
# prediction6 = model.predict(split_test_X)
# print("Accuracy of the Naive Bayes is ", metrics.accuracy_score(prediction6, split_test_Y))

# Random Forests
# model = RandomForestClassifier(n_estimators=100)
# model.fit(split_train_X, split_train_Y)
# prediction7 = model.predict(split_test_X)
# print("Accuracy of the RF is ", metrics.accuracy_score(prediction7, split_test_Y))

# cross validation
# from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
# kfold = KFold(n_splits=10, random_state=22) # k=10
# xyz = []
# accuracy = []
# std = []
# classifiers = ["Linear Svm", "Radial Svm", "Logistic Regression", "KNN", "Decision Tree", "Naive Bayes", "Random Forest"]
# models = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(), 
# KNeighborsClassifier(n_neighbors=3), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(n_estimators=100)]
# for i in models:
#     model = i
#     cv_result = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
#     xyz.append(cv_result.mean())
#     std.append(cv_result.std())
#     accuracy.append(cv_result)
# new_models_dataframe2 = pd.DataFrame({"CV Mean":xyz, "Std":std}, index=classifiers)
# print(new_models_dataframe2)

# plt.subplots(figsize=(12,6))
# box = pd.DataFrame(accuracy, index=[classifiers])
# box.T.boxplot()
# plt.show()

# new_models_dataframe2["CV Mean"].plot.barh(width=0.8)
# plt.title("Average CV Mean Accuracy")
# fig = plt.gcf()
# fig.set_size_inches(8,5)
# plt.show()

# Confusion Matrix
# f,ax=plt.subplots(3,3,figsize=(12,10))
# y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)
# sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
# ax[0,0].set_title('Matrix for rbf-SVM')
# y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
# sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
# ax[0,1].set_title('Matrix for Linear-SVM')
# y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
# sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
# ax[0,2].set_title('Matrix for KNN')
# y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
# sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
# ax[1,0].set_title('Matrix for Random-Forests')
# y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
# sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
# ax[1,1].set_title('Matrix for Logistic Regression')
# y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
# sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
# ax[1,2].set_title('Matrix for Decision Tree')
# y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
# sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
# ax[2,0].set_title('Matrix for Naive Bayes')
# plt.subplots_adjust(hspace=0.2,wspace=0.2)
# plt.show()

# Hyper-Parameters Tuning
# SVM超参数调整
from sklearn.model_selection import GridSearchCV
# C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
# gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# kernel = ["rbf", "linear"]
# hyper = {"kernel":kernel, "C":C, "gamma":gamma}
# gd = GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)
# gd.fit(X, Y)
# print(gd.best_score_)
# print(gd.best_estimator_)
# 最优参数配置
# SVC(C=0.05, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma=0.1, kernel='linear',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)

# Random Forests超参数调整
# n_estimators = range(100, 1000, 100)
# hyper = {"n_estimators":n_estimators}
# gd = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=hyper, verbose=True)
# gd.fit(X, Y)
# print(gd.best_score_)
# print(gd.best_estimator_)
# 最优参数配置
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=900, n_jobs=1,
#             oob_score=False, random_state=0, verbose=0, warm_start=False)

# Ensembling
# 1) Voting Classifier
# from sklearn.ensemble import VotingClassifier
# ensemble_lin_rbf = VotingClassifier(estimators=[("KNN", KNeighborsClassifier(n_neighbors=3)),
# ("RBF", svm.SVC(probability=True, kernel='rbf', C=0.5, gamma=0.1)), 
# ("RFor", RandomForestClassifier(n_estimators=500, random_state=0)), 
# ("LR", LogisticRegression(C=0.05)), 
# ("DT", DecisionTreeClassifier(random_state=0)), 
# ("NB", GaussianNB()), 
# ("svm", svm.SVC(kernel='linear', probability=True)) ], voting="soft").fit(split_train_X, split_train_Y)
# print("the accuracy for ensemble model is:", ensemble_lin_rbf.score(split_test_X, split_test_Y))
# cross = cross_val_score(ensemble_lin_rbf, X, Y, cv=10, scoring="accuracy")
# print("the cross validate score is ", cross.mean())

# 2) Bagging
# 2.1 以KNN为基学习器
from sklearn.ensemble import BaggingClassifier
# model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), random_state=0, n_estimators=700)
# model.fit(split_train_X, split_train_Y)
# prediction = model.predict(split_test_X)
# print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,split_test_Y))
# result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
# print('The cross validated score for bagged KNN is:',result.mean())

# 2.2 以决策树为基学习器
# model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=0, n_estimators=100)
# model.fit(split_train_X, split_train_Y)
# prediction = model.predict(split_test_X)
# print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,split_test_Y))
# result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
# print('The cross validated score for bagged Decision Tree is:',result.mean())

# 3) Boosting
# 3.1 AdaBoost(Adaptive Boosting) 基学习器为决策树
# from sklearn.ensemble import AdaBoostClassifier
# ada = AdaBoostClassifier(n_estimators=200, random_state=0, learning_rate=0.1)
# result = cross_val_score(ada, X, Y, cv=10, scoring="accuracy")
# print('The cross validated score for AdaBoost is:',result.mean())

# 3.2 Stochastic Gradient Boosting 基学习器为决策树
# from sklearn.ensemble import GradientBoostingClassifier
# grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
# result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
# print('The cross validated score for Gradient Boosting is:',result.mean())

# 3.3 XGBoost
# XGBoost有两大类接口：XGBoost原生接口 和 scikit-learn接口
# 此处基于sklearn接口实现xgboost
import xgboost as xgb
xg = xgb.XGBClassifier(n_estimators=900, learning_rate=0.1)
# result = cross_val_score(xg, X, Y, cv=10, scoring='accuracy')
# print('The cross validated score for XGBoost is:',result.mean())
print("train2类型：", type(train2))
print("Y类型：", type(Y))
xg.fit(train2,Y)



# 原始测试集
test2 = dataset[train_len:]
test2.drop(labels=["Survived"], axis=1, inplace=True)

predictions = xg.predict(test2)

# print(train2)
# Y_train = train2["Survived"]
# X_train = train2.drop(labels=["Survived"], axis=1)

# print(dataset.columns.tolist())

# 6.1 cross validate models


# 6.2 Hyperparameter tunning for best models

# 6.3 predict result
# clf = LogisticRegression(penalty='l1', tol=1e-6, C=1.0)
# clf.fit(X_train, Y_train)
# predictions = clf.predict(test2)
result = pd.DataFrame({'PassengerId':IDtest, 'Survived':predictions.astype(np.int32)})
# result.to_csv("LR2_predictions.csv", index=False)
result.to_csv("xgboost_predictions.csv", index=False)


