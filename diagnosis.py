import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV


df = pd.read_csv('brain.csv')
x = df.iloc[:,:-1].astype(float).values
y = df.iloc[:,-1].values

# Feature Engineering
rf = RandomForestClassifier()
rf = SelectFromModel(rd, threshold = 0.10)
rf = rf.fit(x, y)
x_final = rf.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_final, y, test_size=0.20, random_state=1)


pipeline = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(random_state=1)) ])

param_range = [0.01, 0.1, 1.0, 10.0]
param_grid=[{'clf__C': param_range} ]


gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=6)
gs.fit(x_train, y_train)

print('The best accuracy: ', gs.best_score_)
print('The best parameters: ', gs.best_params_)

model = gs.best_estimator_
model.fit(x_train, y_train)

print ('Test accuracy: ', model.score(x_test, y_test))

