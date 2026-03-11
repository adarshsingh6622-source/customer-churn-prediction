import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv("DAta/churn.csv")

# Drop customerID
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing
df = df.fillna(df.mean(numeric_only=True))

# Target
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

# Feature Engineering
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
rf = RandomForestClassifier()

param_grid = {
    "n_estimators":[100,200],
    "max_depth":[5,10]
}

grid = GridSearchCV(rf,param_grid,cv=3)
grid.fit(X_train,y_train)

best_model = grid.best_estimator_

# Prediction
pred = best_model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

# Save model
pickle.dump(best_model,open("model.pkl","wb"))
pickle.dump(scaler,open("scaler.pkl","wb"))
pickle.dump(X.columns,open("features.pkl","wb"))
print("Model saved")