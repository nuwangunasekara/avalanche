# https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
from sklearn.svm import OneClassSVM
X = [[0], [0.44], [0.45], [0.46], [1]]
clf = OneClassSVM(gamma='auto').fit(X)
print('predict\n', clf.predict(X))
yy = clf.fit_predict(X)
print('fit_predict\n', yy)
yy [yy == -1] = 0
print('fit_predict changed\n', yy)
print('score_samples\n', clf.score_samples(X))
# Offset used to define the decision function from the raw scores.
# We have the relation: decision_function = score_samples - offset_.
# The offset is the opposite of intercept_ and is provided for consistency with other
# outlier detection algorithms.
print('offset_\n', clf.offset_)
# Signed distance to the separating hyperplane.
# Signed distance is positive for an inlier and negative for an outlier.
print('decision_function\n', clf.decision_function(X))
print('decision_function(x) = score_samples(x) - offset_\n', clf.score_samples(X)-clf.offset_)
print('sum(decision_function(x))\n', clf.decision_function(X).sum())
print('n_support_\n', clf.n_support_)



# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(random_state=0).fit(X, y)
# clf.predict(X[:2, :])
#
# clf.predict_proba(X[:2, :])


# clf.score(X, y)
