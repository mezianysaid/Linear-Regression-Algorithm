from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd

data=pd.read_excel("pricehouses.xlsx")

X=data.iloc[:,3:7]
Y=data.iloc[:,-1]
# print(X)
# print(Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
print(model.predict(x_test)[5])

















# data=pd.read_excel("pricehouses.xlsx")
# df_x=data.iloc[:,3:7]
# df_y=data.iloc[:,-1]
# print(len(df_x))
# x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)
#
# model=linear_model.LinearRegression()
# model.fit(x_train,y_train)
# print(model.coef_)
# print(model.score(x_test,y_test))
# print(model.predict(x_test)[1])
# print(y_test)

