# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations of gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: A.J.PRANAV
RegisterNumber:  212222230107
*/
```
```
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
```
```
df=pd.read_csv("ex1.txt",header=None)
df.shape
```

```
plt.scatter(df[0],df[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city")
plt.ylabel("Profit")
plt.title("Profit Prediction")
```


```
def computeCost(X,y,theta):
  '''
  take in a numpy array X,y,theta and generate the cost function
  '''
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2

  return 1/(2*m) *np.sum(square_err)

data_n=df.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)
```

```
def gradientDescent(X,y,theta,alpha,num_iters):
  """
  Take in numpy array X, y and theta and update theta by taking num_oters gradient steps
  with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """

  m=len(y)
  J_history = []

  for i in range(num_iters):
      predictions = X.dot(theta)
      error = np.dot(X.transpose(),(predictions-y))
      descent = alpha * 1/m * error
      theta-=descent
      J_history.append(computeCost(X,y,theta))

  return theta, J_history

theta, J_history = gradientDescent(X, y, theta, 0.01, 1500)
print("h(x) ="+str(round(theta[0, 0], 2))+" + "+str(round(theta[1, 0], 2))+"x1")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
```

```
plt.scatter(df[0],df[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
```
```
def predict(x,theta):
   """
    Takes in numpy array of x and theta and return the predicted value of y based on theta
   """

   predictions= np.dot(theta.transpose(),x)

   return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1, 0)))
predict2 = predict(np.array([1, 7]), theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2, 0)))
```
## Output:

### Shape of the dataset
![image](https://github.com/Pranav-AJ/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118904526/f8a5f1ee-4d3d-471a-bfb8-466c620d3399)

### Profit prediction graph
![image](https://github.com/Pranav-AJ/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118904526/20ed8cf6-fd15-4199-8cc5-b2f5caa68be3)

### Compute cost value
![image](https://github.com/Pranav-AJ/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118904526/ff4ecfb6-0fb4-4e6c-95e0-c917e624b096)

### Profit prediction graph
![image](https://github.com/Pranav-AJ/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118904526/396e2bc4-84aa-4972-b3ac-1d4822b299bc)

### Cost function using gradient descent graph
![image](https://github.com/Pranav-AJ/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118904526/a36aff97-68d2-496c-98b2-3308fda89acb)

### Profit for the population of 35000 and 70000
![image](https://github.com/Pranav-AJ/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118904526/128395ac-9db2-461f-ac6c-4b7df6ff13a7)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
