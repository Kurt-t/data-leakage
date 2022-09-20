import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def test_simple_imputer():
    df = pd.DataFrame(data={'col1': [1, np.nan, 2, np.nan], 'col2': [np.nan, 0, np.nan, 1]})
    print(df)
    # sklearn does not support imputing with random numbers
    # it will impute with a fixed randomly generated number (all NaNs imputed with the same number)
    imputer = SimpleImputer(strategy='constant', fill_value=np.random.randint(0, 5))
    df = imputer.fit_transform(df)  # have to fit before transform
    print(df)

def test_column_transform():
    df = pd.DataFrame(data={'Fare': [1, np.nan, 2, np.nan, np.nan],
                            'Age': [np.nan, 0, np.nan, 1, 5],
                            'Survived': [1, 0, 1, 0, 0]})
    print(df)
    train = df[:4]
    test = df[4:]
    y_train = train['Survived']  # split data to X and y before pipeline since pipeline doesn't preserve df
    X_train = train.drop('Survived', axis = 1)
    X_test = test.drop('Survived', axis = 1)    
    age_avg = train['Age'].mean()
    age_std = train['Age'].std()
    imputer = ColumnTransformer([('fare_imputer', SimpleImputer(strategy='mean'), ['Fare']),
        ('age_imputer', SimpleImputer(strategy='constant', fill_value=np.random.randint(age_avg - age_std, age_avg + age_std)), ['Age'])],  # using a constant that's relevant to column stat
        remainder='passthrough')
    pipe = Pipeline([('imputer', imputer)])
    pipe.fit(X_train)
    X_train = pipe.transform(X_train)
    X_test = pipe.transform(X_test)
    print(X_train)
    print(X_test)
    print(y_train)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)

# test if fit_transform() then transform() works
def test_fit_transform():
    X = pd.DataFrame(data={'col1': [1, 2, 3, 4, 5]})
    X_train = X[:3]
    X_test = X[3:]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    pipe = Pipeline([('sc', sc)])
    X_train = pipe.fit_transform(X_train)
    X_test = pipe.transform(X_test)
    print(X_train)
    print(X_test)

test_fit_transform()
print("complete")