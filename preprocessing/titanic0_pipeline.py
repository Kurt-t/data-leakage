import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.head()
test.head()


data = pd.concat([train, test], sort=False)
data.head()
print(len(train), len(test), len(data))

data.isnull().sum()

data['Sex'].replace(['male','female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train = data[:len(train)]
test = data[len(train):]
y_train = train['Survived']  # split data to X and y before pipeline since pipeline doesn't preserve df
X_train = train.drop('Survived', axis = 1)
X_test = test.drop('Survived', axis = 1)

age_avg = train['Age'].mean()
age_std = train['Age'].std()
imputer = ColumnTransformer([('fare_imputer', SimpleImputer(strategy='mean'), ['Fare']),
    ('age_imputer', SimpleImputer(strategy='constant', fill_value=np.random.randint(age_avg - age_std, age_avg + age_std)), ['Age'])],  # using a constant that's relevant to column stat
    remainder='passthrough')
# SimpleImputer parameter missing_values: could be nan or -1, might need insight of the dataset, not just the notebook code

pipe = Pipeline([('imputer', imputer)])
pipe.fit(X_train)
X_train = pipe.transform(X_train)  # output is an ndarray, not df
X_test = pipe.transform(X_test)

# delete X_train.head() since it's not a df
# another way is to get back the column names after pipeline:
# X_cols = X_train.columns.tolist()
# ...pipelining
# X_train = pd.DataFrame(X_train, columns= X_cols)
# X_test = pd.DataFrame(X_test, columns= X_cols)
# ref: https://stackoverflow.com/questions/70993316/get-feature-names-after-sklearn-pipeline

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)