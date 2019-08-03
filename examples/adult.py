from corels import *
from metrics import ConfusionMatrix, Metric
import pandas as pd

X, y, features, prediction = load_from_csv("data/adult_train_binary.csv")
X_test, y_test, features_test, prediction_test = load_from_csv("data/adult_test_binary.csv")

"""
print(features)

for idx, val in enumerate(features):
    print("{} ------ {}".format(idx + 1, val))

"""
#print(prediction)


c = CorelsClassifier(n_iter=1000000, c=0.005, max_card=1, policy="lower_bound", verbosity=["progress"], beta=0.1, fairness=1, maj_pos=20, min_pos=19)

c.fit(X, y, features=features, prediction_name="(income:>50K)")


print("-------------------------- Learned rulelist ------------------------------")
print(c.rl())


dataset = pd.read_csv("data/adult_test_binary.csv")

dataset["predictions"] = c.predict(X_test)


cm = ConfusionMatrix(dataset["gender:Female"], dataset["gender:Male"], dataset["predictions"], dataset["income"])
cm_minority, cm_majority = cm.get_matrix()
fm = Metric(cm_minority, cm_majority)

print("Train accuracy: {}".format(c.score(X, y)))

print("Test accuracy {}".format(c.score(X_test, y_test)))

print("=========> Statiscal parity {}".format(fm.statistical_parity()))
print("=========> Predictive parity {}".format(fm.predictive_parity()))
print("=========> Predictive equality {}".format(fm.predictive_equality()))
print("=========> Equal opportunity {}".format(fm.equal_opportunity()))


