import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess_input

categories = {
    "low": 0,
    "medium": 1,
    "high": 2
}

train_df = pd.read_csv("train_cleaned.csv")

train_df["GPA_class"] = pd.cut(
    train_df["GPA"],
    bins=[-float("inf"), 3.0, 3.5, float("inf")],
    labels=[0, 1, 2],
    right=True,
    include_lowest=True
).astype(int)

X_train = train_df.drop(columns=["GPA", "GPA_class"])
y_train = train_df["GPA_class"]

try:
    feedback_df = pd.read_csv("feedback_data.csv")

    feedback_rows = []
    feedback_labels = []

    for _, row in feedback_df.iterrows():
        raw_input = row.drop("true_label").to_dict()
        X_new = preprocess_input(raw_input)
        feedback_rows.append(X_new)
        feedback_labels.append(categories[row["true_label"]])

    if feedback_rows:
        X_feedback = pd.concat(feedback_rows, ignore_index=True)
        y_feedback = pd.Series(feedback_labels)

        X_train = pd.concat([X_train, X_feedback], ignore_index=True)
        y_train = pd.concat([y_train, y_feedback], ignore_index=True)

except FileNotFoundError:
    print("No feedback_data.csv found. Retraining on original training data only.")

model = LogisticRegression(
    solver="lbfgs",
    C=1,
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)

joblib.dump(model, "final_model.pkl")
print("Retrained model saved to final_model.pkl")
