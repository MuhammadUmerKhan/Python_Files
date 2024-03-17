import math
import pandas as pd
from sklearn import linear_model
import numpy as np

def gradient_descebt(x, y):
    m_curr = b_curr = 0
    iteration = 10000
    learning_rate = 0.08
    n = len(x)
    for i in range(iteration):
        y_predicted = m_curr * x + b_curr;
        cost = (1/n) * sum([j**2 for j in (y - y_predicted)])
        md = -(2/n) * sum(x*(y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
#  y = 2x + 3
gradient_descebt(x, y)

# ___________________

def test_score():
    df = pd.read_csv("test_scores.csv")
    df.head()

    X = df[['math']]
    Y = df.cs
    lr = linear_model.LinearRegression()
    lr.fit(X, Y)
    return lr.coef_, lr.intercept_
# test_score()

def test_score_gradient_desent(x, y):
    m_curr = b_curr = 0
    iteration = 1000000
    learning_rate = 0.0002
    n = len(x)
    cost_previous = 0
    for i in range(iteration):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break;
        cost_previous = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))
        
    return m_curr, b_curr
if __name__ == '__main__':
    # df = pd.read_csv("")
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)
    
    m, b = test_score_gradient_desent(x, y)
    # print("Using gradient descent function: Coef {} Intercept {}".format(m, b))
    m_sklearn, b_sklearn = test_score()
    # print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))
    
df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\All Files\ChurnData.csv")
df