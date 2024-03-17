import pickle as pk
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("bengaluru_house_prices_no.csv")
df.head()

dummies = pd.get_dummies(df['location'])
dummies = dummies.astype('int')

df = pd.concat([df.drop(columns='location', axis='columns'),  dummies.drop(columns='other', axis='columns')], axis='columns')
df.head()

# Build a model Now......
X = df.drop(columns='price')
Y = df.price

X.head()
Y.head()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
lr = LinearRegression()
lr.fit(x_train, y_train)
lr.score(x_test, y_test) # 0.8629132245229446

# use k-fold cross validation
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
score = cross_val_score(LinearRegression(), X, Y, cv=cv)
score.mean() # 0.8477957812447761



def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
find_best_model_using_gridsearchcv(X,Y)

# Test the model for few properties
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    
    return lr.predict([x])[0]

predict_price('1st Phase JP Nagar',1000, 2, 2)
predict_price('1st Phase JP Nagar',1000, 3, 3)
predict_price('Indira Nagar',1000, 2, 2)
predict_price('Indira Nagar',1000, 3, 3)

with open('banglore_home_prices_model.pickle', 'wb') as f:
    pk.dump(lr, f)

columns = {
    'data_columns' : [col.lower() for col in X.columns]
}

with open("columns.json","w") as f:
    f.write(json.dumps(columns))