import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp  

data=pd.read_csv('Salary.csv')
print(data.sample(10))

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 1/3, random_state=0)  

from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(X_train, y_train) 


y_pred= regressor.predict(X_test)  
X_pred= regressor.predict(X_train)  
mtp.scatter(X_train, y_train, color="green")   
mtp.plot(X_train, X_pred, color="red")    
mtp.title("Salary vs Experience (Training Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()  

#visualizing the Test set results  
mtp.scatter(X_test, y_test, color="blue")   
mtp.plot(X_train, X_pred, color="red")    
mtp.title("Salary vs Experience (Test Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()
