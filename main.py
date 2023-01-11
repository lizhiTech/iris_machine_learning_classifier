# 引入库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

# 加载数据集
iris_dataset = load_iris()

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'])

# 使用不同的分类器（均使用默认参数）

# KNN
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
print('KNN accuracy:', clf.score(X_test, y_test))

# 逻辑回归
clf = LogisticRegression()
clf.fit(X_train, y_train)
print('逻辑回归 accuracy:', clf.score(X_test, y_test))

# 决策树
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print('决策树 accuracy:', clf.score(X_test, y_test))

# 梯度提升
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
print('梯度提升 accuracy:', clf.score(X_test, y_test))

# AdaBoost
clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
print('AdaBoost accuracy:', clf.score(X_test, y_test))

# 随机森林
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print('随机森林 accuracy:', clf.score(X_test, y_test))

# 高斯朴素贝叶斯
clf = GaussianNB()
clf.fit(X_train, y_train)
print('高斯朴素贝叶斯 accuracy:', clf.score(X_test, y_test))

# 多项式朴素贝叶斯
clf = MultinomialNB()
clf.fit(X_train, y_train)
print('多项式朴素贝叶斯 accuracy:', clf.score(X_test, y_test))

# 线性判别分析
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
print('线性判别分析 accuracy:', clf.score(X_test, y_test))

# 二次判别分析
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)
print('二次判别分析 accuracy:', clf.score(X_test, y_test))

# 支持向量机
clf = SVC()
clf.fit(X_train, y_train)
print('支持向量机 accuracy:', clf.score(X_test, y_test))