# Import the iris dataset
from sklearn import datasets
import random
import xlrd
import pymysql

# connection = pymysql.connect(host='localhost', user='root', password='890xyz', db='PostureAlert')
# cursor=connection.cursor()

features = []
labels = []
X_train = []
y_train = []
X_test = []
y_test = []
readingID = []

def setPosture(predictions):
    global readingID
    # print(readingID , predictions)
    for i in range(len(predictions)):
    
        connection = pymysql.connect(host='localhost', user='root', password='890xyz', db='PostureAlert')
        cursor=connection.cursor()
        print("UPDATE SensorReadings SET Posture = %s WHERE ReadingID = %s", str(predictions[i]), str(readingID[i]))
        sql=("UPDATE SensorReadings SET Posture = %s WHERE ReadingID = %s;")
        cursor.execute(sql, (str(predictions[i]), str(readingID[i])))
        connection.commit()
        print('changed', cursor.rowcount)
        cursor.close()
        connection.close()


 

# -------------------------------------------   -----------------------------------------------------------------------------------
def getTrainDataset():
    # print("1...................");
    global features, labels, X_train, y_train
    features = []
    labels = []
    # trainning set
    connection = pymysql.connect(host='localhost', user='root', password='12345678', db='PostureAlert')
    cursor=connection.cursor()
    sql=("SELECT * FROM SensorReadings WHERE Posture IS NOT NULL")
    cursor.execute(sql)
    data=cursor.fetchall()
    # print(data)
    cursor.close()
    connection.close()

    for row in range(len(data)):
        features.append(data[row][1:9])
        labels.append(data[row][9])

    X_train = features
    y_train = labels
    # print(X_train)
    # print(y_train)
    print('Train sample size = ', len(X_train), len(y_train))


# ------------------------------------------------------------------------------------------------------------------------------
def getNewReadings():
    # print("2...................");
    global features, labels, X_test, y_test, readingID
    features = []
    labels = []
    readingID = []
    # new data set
    connection = pymysql.connect(host='localhost', user='root', password='12345678', db='PostureAlert')
    cursor=connection.cursor()
    sql=("SELECT * FROM SensorReadings WHERE Posture IS NULL")
    cursor.execute(sql)
    data=cursor.fetchall()
    # print(data)
    cursor.close()
    connection.close()

    for row in range(len(data)):
        features.append(data[row][1:9])
        readingID.append(data[row][0])

    X_test = features
    y_test = labels
    # print(X_test)
    # print(y_test)  
    print('Test sample size = ', len(X_test), len(y_test))

# ------------------------------------------------------------------------------------------------------------------------------

def measure(classifier, outputLabel):
    # print("3...................");
    # train
    classifier.fit(X_train, y_train)

    # new sample
    predictions = classifier.predict(X_test)
    # print(outputLabel + " Accuracy: " + str(metrics.accuracy_score(y_test, predictions)))
    # print(predictions)
    return predictions

import random;
class RandomClassifier():
    # print("4...................");
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            predictions.append(
                random.choice(self.y_train)
            )
        return predictions

from scipy import spatial
from collections import deque, Counter
class KNNClassifier(RandomClassifier):
    # print("5...................");
    def __init__(self, k):
        self.k = k

    @staticmethod
    def distance(a, b):
        return spatial.distance.euclidean(a, b)

    def _closest(self, row, k=1):
        distances = {};
        for i in range(0, len(self.X_train)):
            dist = KNNClassifier.distance(row, self.X_train[i])
            label = self.y_train[i]
            try:
                distances[dist].append(label)
            except KeyError:
                distances[dist] = [label]
        min_dists = deque(sorted(distances.keys()))
        min_dist_labels = []
        while (len(min_dist_labels) < k):
            min_dist_labels.extend(distances[min_dists.popleft()])
        frequencies = Counter(min_dist_labels).most_common()
        min_labels = []
        max_frequency = max([c[1] for c in frequencies])
        for f in (f for f in frequencies if f[1] == max_frequency):
            min_labels.append(f[0])
        return random.choice(min_labels)
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            predictions.append(
                self._closest(row, self.k)
            )
        return predictions


class NNClassifier(KNNClassifier):
    def __init__(self):
        self.k = 1

import time
from sklearn import metrics, tree, neighbors
# print("7...................");
algos = {
            # 'Tree': tree.DecisionTreeClassifier(),
            # 'SKLearn K-Nearest Neighbor': neighbors.KNeighborsClassifier(),
            # 'Random': RandomClassifier(),
            # 'Nearest Neighbor': NNClassifier(),
            '3-Nearest Neighbor': KNNClassifier(3),
            # '7-Nearest Neighbor': KNNClassifier(7),
            # '15-Nearest Neighbor': KNNClassifier(15),
        }

# print("8...................");
while (1):
    for outputLabel, algo in algos.items():
        getTrainDataset()
        getNewReadings()
        print("Testing ", outputLabel)
        # predictions = measure(algo, outputLabel)
        # setPosture(predictions)
    time.sleep(10)



