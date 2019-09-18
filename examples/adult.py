from faircorels import *
from metrics import ConfusionMatrix, Metric
import pandas as pd

X, y, features, prediction = load_from_csv("data/adult_train_binary.csv")
X_test, y_test, features_test, prediction_test = load_from_csv("data/adult_test_binary.csv")


c = CorelsClassifier(random_state=220, n_iter=150000, c=0.0001, max_card=1, policy="bfs", bfs_mode=2, useUnfairnessLB=True, fairness=1, maj_pos=20, min_pos=19, epsilon=0.00, mode=3)

c.fit(X, y, performGeomR=0, initNBNodes=10000, geomRReason=1.25, features=features, prediction_name="(income:>50K)")


print("-------------------------- Learned rulelist ------------------------------")
print(c.rl())


dataset = pd.read_csv("data/adult_train_binary.csv")

dataset["predictions"] = c.predict(X)


cm = ConfusionMatrix(dataset["gender:Female"], dataset["gender:Male"], dataset["predictions"], dataset["income"])
cm_minority, cm_majority = cm.get_matrix()
fm = Metric(cm_minority, cm_majority)

print("Train accuracy: {}".format(c.score(X, y)))

print("Test accuracy {}".format(c.score(X_test, y_test)))

print("=========> Statiscal parity {}".format(fm.statistical_parity()))
print("=========> Predictive parity {}".format(fm.predictive_parity()))
print("=========> Predictive equality {}".format(fm.predictive_equality()))
print("=========> Equal opportunity {}".format(fm.equal_opportunity()))