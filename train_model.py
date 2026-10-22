import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

train_df = pd.read_csv("train_cleaned.csv")
val_df = pd.read_csv("val_cleaned.csv")
test_df = pd.read_csv("test_cleaned.csv")

for df in [train_df, val_df, test_df]:
    df["GPA_class"] = pd.cut(
        df["GPA"],
        bins=[-float("inf"), 3.0, 3.5, float("inf")],
        labels=[0, 1, 2],
        right=True,
        include_lowest=True
    ).astype(int)

X_train = train_df.drop(columns=["GPA", "GPA_class"])
y_train = train_df["GPA_class"]

X_val = val_df.drop(columns=["GPA", "GPA_class"])
y_val = val_df["GPA_class"]

X_test = test_df.drop(columns=["GPA", "GPA_class"])
y_test = test_df["GPA_class"]

model = LogisticRegression(
    solver="lbfgs",
    C=1,
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Weighted F1:", f1_score(y_val, y_val_pred, average="weighted"))

y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Weighted F1:", f1_score(y_test, y_test_pred, average="weighted"))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

joblib.dump(model, "final_model.pkl")
joblib.dump(list(X_train.columns), "feature_columns.pkl")

print("\nSaved final_model.pkl and feature_columns.pkl")
