import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

titanic_url = ('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
data = pd.read_csv(titanic_url)
X = data.drop('survived', axis=1)
y = data['survived']

# SimpleImputer on string is not available for string in ONNX-ML specifications.
# So we do it beforehand.
for cat in ['embarked', 'sex', 'pclass']:
    X[cat].fillna('missing', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    # --- SimpleImputer is not available for strings in ONNX-ML specifications.
    # ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

clf.fit(X_train, y_train)

from skl2onnx.common.data_types import FloatTensorType, StringTensorType, Int64TensorType


def convert_dataframe_schema(df, drop=None):
    inputs = []
    for k, v in zip(df.columns, df.dtypes):
        if drop is not None and k in drop:
            continue
        if v == 'int64':
            t = Int64TensorType([None, 1])
        elif v == 'float64':
            t = FloatTensorType([None, 1])
        else:
            t = StringTensorType([None, 1])
        inputs.append((k, t))
    return inputs


inputs = convert_dataframe_schema(X_train)

import pprint

pprint.pprint(inputs)

from skl2onnx import convert_sklearn

to_drop = {'parch', 'sibsp', 'cabin', 'ticket', 'name', 'body', 'home.dest', 'boat'}
inputs = convert_dataframe_schema(X_train, to_drop)
try:
    model_onnx = convert_sklearn(clf, 'pipeline_titanic', inputs)
except Exception as e:
    print(e)

X_train['pclass'] = X_train['pclass'].astype(str)
X_test['pclass'] = X_test['pclass'].astype(str)
inputs = convert_dataframe_schema(X_train, to_drop)

model_onnx = convert_sklearn(clf, 'pipeline_titanic', inputs)

# And save.
with open("pipeline_titanic.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

print("predict", clf.predict(X_test[:5]))
print("predict_proba", clf.predict_proba(X_test[:1]))

X_test2 = X_test.drop(to_drop, axis=1)
inputs = {c: X_test2[c].values[:5] for c in X_test2.columns}
for c in numeric_features:
    inputs[c] = inputs[c].astype(np.float32)
for k in inputs:
    inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))

import onnxruntime as rt

sess = rt.InferenceSession("pipeline_titanic.onnx")
print(inputs)
pred_onx = sess.run(None, inputs)
print("predict: 1", pred_onx[0][:5])
print("predict_proba 1", pred_onx[1][:5])

