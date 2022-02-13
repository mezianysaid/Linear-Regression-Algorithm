from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_boston

boston=load_boston()
X=pd.DataFrame(boston.data,columns=boston.feature_names)
Y=pd.DataFrame(boston.target)
# print(X)
# print(Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=4)
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)[3]
sc=model.score(x_test,y_test)
print(sc)