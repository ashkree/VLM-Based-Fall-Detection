import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

results = pd.read_csv("./results/gpt_results.csv")

labels = ["FALL", "NO_FALL"]

predicted = results["predicted_label"]
actual = results["actual_label"]

print(confusion_matrix(actual, predicted, labels=labels))
print(classification_report(actual, predicted, labels=labels))
