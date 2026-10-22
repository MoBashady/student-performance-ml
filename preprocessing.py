import pandas as pd
import joblib

feature_columns = joblib.load("feature_columns.pkl")
feature_means = joblib.load("feature_means.pkl")
feature_stds = joblib.load("feature_stds.pkl")
numeric_columns = set(joblib.load("numeric_columns.pkl"))

numerical_features = {
    "age": "Age",
    "grade": "Grade",
    "ses_quartile": "SES_Quartile",
    "math_score": "TestScore_Math",
    "reading_score": "TestScore_Reading",
    "science_score": "TestScore_Science",
    "attendance_rate": "AttendanceRate",
    "study_hours": "StudyHours",
    "free_time": "FreeTime",
    "go_out": "GoOut",
}

binary_features = {
    "internet_access": "InternetAccess",
    "extracurricular": "Extracurricular",
    "part_time_job": "PartTimeJob",
    "romantic": "Romantic",
    "parent_support": "ParentSupport",
}

categorical_features = {
    "gender": "Gender",
    "race": "Race",
    "school_type": "SchoolType",
    "parental_education": "ParentalEducation",
    "locale": "Locale",
}


def normalize(x, mean, std):
    if std == 0:
        return 0.0
    return (x - mean) / std


def preprocess_input(user_input: dict) -> pd.DataFrame:
    row = {col: 0.0 for col in feature_columns}

    for key, model_col in numerical_features.items():
        if key in user_input:
            value = float(user_input[key])
            if model_col in numeric_columns:
                row[model_col] = normalize(
                    value,
                    feature_means[model_col],
                    feature_stds[model_col]
                )
            else:
                row[model_col] = value

    for key, model_col in binary_features.items():
        if key in user_input:
            value = 1.0 if user_input[key] else 0.0
            if model_col in numeric_columns:
                row[model_col] = normalize(
                    value,
                    feature_means[model_col],
                    feature_stds[model_col]
                )
            else:
                row[model_col] = value

    for key, prefix in categorical_features.items():
        if key in user_input:
            value = str(user_input[key])
            col_name = f"{prefix}_{value}"
            if col_name in row:
                row[col_name] = 1.0

    return pd.DataFrame(
        [[row[col] for col in feature_columns]],
        columns=feature_columns
    )


if __name__ == "__main__":
    sample_input = {
        "age": 17,
        "grade": 12,
        "ses_quartile": 3,
        "math_score": 80,
        "reading_score": 78,
        "science_score": 74,
        "attendance_rate": 0.90,
        "study_hours": 2.5,
        "internet_access": True,
        "extracurricular": True,
        "part_time_job": False,
        "parent_support": True,
        "romantic": False,
        "free_time": 3,
        "go_out": 2,
        "gender": "Female",
        "race": "White",
        "school_type": "Public",
        "parental_education": "SomeCollege",
        "locale": "Suburban"
    }

    X_new = preprocess_input(sample_input)

    print("Preprocessed shape:", X_new.shape)
    print("\nFirst row:")
    print(X_new.head())

    print("\nColumn order matches feature_columns:", list(X_new.columns) == feature_columns)

    model = joblib.load("final_model.pkl")
    pred = model.predict(X_new)[0]
    probs = model.predict_proba(X_new)[0]

    print("\nPredicted class:", pred)
    print("Probabilities:", probs)
