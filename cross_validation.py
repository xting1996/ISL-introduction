
# coding: utf-8
#module
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# read in the iris data
iris = load_iris()
X = iris.data
y = iris.target

for i in range(1,5):
    print("random_state is",i,",and accuarcy score is:")
    
    X_train ,X_test,y_train,y_test = train_test_split(X,y,random_state = i)
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print(metrics.accuracy_score(y_test,y_pred))
    
    
####
# y_test是有label的，y_pred是预测出来的label
#  不同的训练集，测试集分割的方法导致其准确率不同。
#  交叉验证的思想即：将数据集进行一系列分割，生成一组不同的训练测试集，
#  然后分别训练模型并计算测试准确率，
#  最后对结果进行平均处理
#高方差估计


# k折交叉验证
#k：将数据分为K份，演示k-fold交叉验证是如何进行数据分割的
# simulate splitting a dataset of 25 observations into 5 folds
from sklearn.cross_validation import KFold
kf = KFold(25,n_folds = 5,shuffle = False)
print('{}{:^61}{}'.format('Iteration','Training set observations','Testing set observations'))
for iteration , data in enumerate(kf,start = 1):
    print('{:^9}{}{}'.format(iteration, data[0], data[1]))


from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors = 5)
scores = cross_val_score(knn,X,y,cv = 10,scoring = 'accuracy')
print(scores)
#计算十次迭代计算平均的测试准确率
#准确率可以用于调节参数
print(scores.mean())


# search for an optimal value of k for KNN model
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn,X,y,cv = 10,scoring = 'accuracy')
    k_scores.append(scores.mean())
print(k_scores)
get_ipython().magic('matplotlib inline')
plt.plot(k_range,k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross validated accuracy")

#这样能看出n_neighbors=k中的k取值为何时。


#用于模型选择
#KNN
knn = KNeighborsClassifier(n_neighbors = 20)
print(cross_val_score(knn,X,y,cv = 10,scoring = 'accuracy').mean())

#Logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg,X,y,cv = 10,scoring = 'accuracy').mean())

#0.98
#0.953333333333
#结果表明 KNN 这个model更好


# 用于特征选择  feature selection
# 主要方法是增加或者减少特征，以此来查看应该选择多少特征值
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# read the adverstisingv dataset 
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
# three features
feature_cols = ['TV','radio','newspaper']
#从dataframe中选择有这三个特征的数据值
X = data[feature_cols]
y = data.sales

# LinearRegression
lm = LinearRegression()
scores = cross_val_score(lm,X,y,cv = 10,scoring = 'mean_squared_error')

# 此处的scores为负值，因为mean_squared_error是一种损失函数，优化的目标是使其最小化，而分类准确率是一种奖励函数胡，优化的目标是使其最大化
# convert
mse_scores = -scores
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores.mean())

# two features
feature_cols =  ['TV','radio']
X = data[feature_cols]
print(np.sqrt(-cross_val_score(lm,X,y,cv = 10,scoring = 'mean_squared_error')).mean())
 
# 1.69135317081
# 1.67967484191
#所以使用三个feature更好

