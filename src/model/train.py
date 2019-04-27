from sklearn import datasets, svm

iris = datasets.load_iris()
iris.target_names

X = iris.data
y = iris.target

X.shape
y.shape  # 150 samples, 4 features

clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X, y)
clf.predict(X[-1:])

