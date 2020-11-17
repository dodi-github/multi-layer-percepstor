Python 3.8.6 (tags/v3.8.6:db45529, Sep 23 2020, 15:52:53) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from sklearn.neural_network import MLPClassifier
>>> from sklearn.linear_model import Perceptron
>>> import sklearn.metrics as metric
>>> import numpy as np
>>> X_training=[[1, 1],
	    [1, 0],
	    [0, 1],
	    [0, 0]
	    ]
>>> y_training=[1,
	    1,
	    1,
	    0
	    ]
>>> X_testing=X_training
>>> y_true=y_training
>>> ptn = Perceptron(max_iter=500)
>>> ptn.fit(X_training, y_training)
Perceptron(max_iter=500)
>>> y_pred=ptn.predict(X_testing)
>>> print(y_pred)
[1 1 1 0]
>>> accuracy=metric.accuracy_score(y_true, y_pred, normalize=True)
>>> print('acuracy=',accuracy)
acuracy= 1.0
>>> print(ptn.intercept_, ptn.coef_)
[-1.] [[2. 2.]]
>>> mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(1,1), activation='logistic')
>>> mlp.fit(X_training, y_training)
MLPClassifier(activation='logistic', hidden_layer_sizes=(1, 1), solver='lbfgs')
>>> y_pred=mlp.predict(X_testing)
>>> print(y_pred)
[1 1 1 0]
>>> accuracy=metric.accuracy_score(np.array(y_true).flatten(), np.array(y_pred).flatten(), normalize=True)
>>> print('acuracy=',accuracy)
acuracy= 1.0
>>> print([coef.shape for coef in mlp.coefs_])
[(2, 1), (1, 1), (1, 1)]
>>> mlp.coefs_
[array([[3.96343766],
       [3.97634455]]), array([[7.88148794]]), array([[14.84702799]])]
>>> 