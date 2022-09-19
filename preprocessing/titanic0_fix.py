import numpy as np
import pandas as pd

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

train['Fare'].fillna(np.mean(train['Fare']), inplace=True)
test['Fare'].fillna(np.mean(train['Fare']), inplace=True)

age_avg = train['Age'].mean()
age_std = train['Age'].std()
train['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
test['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)


y_train = train['Survived']
X_train = train.drop('Survived', axis = 1)
X_test = test.drop('Survived', axis = 1)


X_train.head()
y_train.head()


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)