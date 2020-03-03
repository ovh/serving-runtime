from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression

clr = LogisticRegression()
clr.fit(X_train, y_train)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

import onnxruntime as rt

sess = rt.InferenceSession("iris.onnx")

print("input name='{}' and shape={}".format(
    sess.get_inputs()[0].name, sess.get_inputs()[0].shape))
print("output name='{}' and shape={}".format(
    sess.get_outputs()[0].name, sess.get_outputs()[0].shape))

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

import numpy

pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[:5]
print(X_test.astype(numpy.float32)[:5])
print("Classification:")
print(pred_onx[:5])

prob_name = sess.get_outputs()[1].name
prob_rt = sess.run([prob_name], {input_name: X_test.astype(numpy.float32)})[:5]

import pprint

print("Proba")
pprint.pprint(prob_rt[0][:5])
