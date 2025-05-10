import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

data = pd.DataFrame({'transaction_id': [1, 2, 3, 4, 5],
                     'amount': [100, 5000, 200, 10000, 50],
                     'user_id': [101, 102, 103, 101, 104],
                     'timestamp': ['2025-05-01', '2025-05-02', '2025-05-02', '2025-05-03', '2025-05-03']})
data = data.drop(columns=['transaction_id', 'timestamp'])

model = IsolationForest(contamination=0.2)
data['fraud_score'] = model.fit_predict(data[['amount']])
true_labels = [0, 0, 0, 1, 0]
print(classification_report(true_labels, data['fraud_score'].apply(lambda x: 1 if x == -1 else 0)))
