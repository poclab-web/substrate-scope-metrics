from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def loo_knn(x, y, k, metric="euclidean"):
    """Classifies by k-nearest neighbor algorithm using leave-one-out cross-validation.

    Args:
        x (numpy.ndarray): Unstandardized explanatory variables.
        y (numpy.ndarray): Objective variable (1 dimension).
        k (int): The value of k of kNN.
        metric (str, optional): Definition of distance used in kNN. Defaults to "euclidean".

    Returns:
        tuple: Prediction results (list), confusion matrix (numpy.ndarray), accuracy, precision, recall, f1
    """
    y_true = []
    y_pred = []

    loo = LeaveOneOut()
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

    for train_index, test_index in loo.split(x):
        x_train = x[train_index]
        y_train = y[train_index]

        x_test = x[test_index]
        y_test = y[test_index]

        knn.fit(x_train, y_train)
        result = knn.predict(x_test)

        y_true.append(y_test[0])
        y_pred.append(result[0])

    cm = metrics.confusion_matrix(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)

    return y_pred, cm, accuracy, precision, recall, f1


def loo_std_knn(x, y, k, metric="euclidean"):
    """Standardizes input data and Classifies by k-nearest neighbor algorithm using leave-one-out cross-validation.

    Args:
        x (numpy.ndarray): Unstandardized explanatory variables.
        y (numpy.ndarray): Objective variable (1 dimension).
        k (int): The value of k of kNN.
        metric (str, optional): Definition of distance used in kNN. Defaults to "euclidean".

    Returns:
        tuple: Prediction results (list), confusion matrix (numpy.ndarray), accuracy, precision, recall, f1
    """
    y_true = []
    y_pred = []

    scaler = StandardScaler()
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    loo = LeaveOneOut()

    pipe = Pipeline([("scaler", scaler), ("knn", knn)])

    for train_index, test_index in loo.split(x):
        x_train = x[train_index]
        y_train = y[train_index]

        x_test = x[test_index]
        y_test = y[test_index]

        pipe.fit(x_train, y_train)
        result = pipe.predict(x_test)

        y_true.append(y_test[0])
        y_pred.append(result[0])

    cm = metrics.confusion_matrix(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)

    return y_pred, cm, accuracy, precision, recall, f1 
