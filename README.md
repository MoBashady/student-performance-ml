# Student Performance Prediction System

A machine learning application for predicting student performance categories using Python and scikit-learn.

The project covers the complete machine learning workflow, including preprocessing, model training, prediction through a Flask REST API, a desktop GUI client, user feedback collection, and model retraining.

## Features

- Machine learning classification for student performance prediction
- Flask REST API for model inference
- Desktop GUI for entering student information and viewing predictions
- Prediction probabilities for each performance category
- Reusable preprocessing pipeline
- User feedback collection
- Automatic model retraining using corrected feedback
- Separation between model training, inference, preprocessing, and retraining

## Project Structure

```text
student-performance-ml/
│
├── app.py
├── gui_client.py
├── preprocessing.py
├── train_model.py
├── retrain_model.py
├── numeric_columns.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

### `app.py`

Runs the Flask backend responsible for serving predictions.

The API provides functionality for:

- checking whether the server is running
- receiving student data
- preprocessing inputs
- generating model predictions
- returning predicted performance categories
- returning prediction probabilities
- receiving corrected user feedback

### `gui_client.py`

Provides a desktop interface for interacting with the prediction system.

The user can enter student information, submit it to the Flask API, and view the predicted performance category and associated probabilities.

### `preprocessing.py`

Handles preprocessing of incoming student data before it is passed to the machine learning model.

This includes:

- numerical feature processing
- categorical feature encoding
- maintaining the expected feature ordering
- applying stored preprocessing information

### `train_model.py`

Handles training of the machine learning model using the cleaned training, validation, and test datasets.

The final system uses a Logistic Regression classifier for student performance prediction.

### `retrain_model.py`

Supports retraining the model using corrected user feedback.

The application stores feedback and can incorporate newly corrected samples into the training process so that the model can be updated over time.

## Machine Learning Workflow

```text
Student Data
     |
     v
Preprocessing
     |
     v
Logistic Regression Model
     |
     v
Prediction + Probabilities
     |
     v
Flask REST API
     |
     v
Desktop GUI
     |
     v
User Feedback
     |
     v
Model Retraining
```

## Model Development

During development, multiple regression and classification algorithms were evaluated before selecting the final classification model.

The project evaluated **17 machine learning models** across regression and classification tasks using a dataset containing approximately **10 million records**.

Evaluation included metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- R²
- MAE
- MSE
- RMSE

After model comparison and tuning, Logistic Regression was selected for the deployed classification system because it achieved competitive performance while remaining relatively simple and efficient.

## REST API

The Flask backend exposes endpoints for interacting with the model.

### Health Check

```text
GET /health
```

Used to verify that the prediction service is running.

### Prediction

```text
POST /predict
```

Accepts student information and returns:

- predicted performance category
- probability values for each category

### Feedback

```text
POST /feedback
```

Accepts corrected user feedback that can later be used for retraining.

## Feedback and Retraining

The system includes a feedback mechanism that allows incorrect predictions to be corrected.

Feedback is stored locally and can later be combined with the original training data.

The retraining pipeline is designed to retrain the model after every **5 new feedback samples**, allowing the system to incorporate corrected examples over time.

## Requirements

Install the Python dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Train the Model

The training script expects cleaned datasets such as:

```text
train_cleaned.csv
val_cleaned.csv
test_cleaned.csv
```

Run:

```bash
python train_model.py
```

This generates the trained model and preprocessing artifacts required by the application.

### 2. Start the Flask API

```bash
python app.py
```

### 3. Start the GUI

In another terminal:

```bash
python gui_client.py
```

The GUI will communicate with the Flask backend to generate predictions.

## Model Artifacts

The application expects generated files such as:

```text
final_model.pkl
feature_columns.pkl
feature_means.pkl
feature_stds.pkl
numeric_columns.pkl
```

`numeric_columns.pkl` is included in this repository.

The remaining files are generated during the model-training process and are intentionally excluded from Git because they are derived artifacts.

## Technologies

- Python
- scikit-learn
- Flask
- REST APIs
- pandas
- NumPy
- Tkinter
- Pickle

## What I Learned

This project gave me practical experience with:

- Large-scale machine learning workflows
- Regression and classification model comparison
- Model evaluation and hyperparameter tuning
- Data preprocessing
- Logistic Regression
- Flask REST API development
- Connecting a GUI to a machine learning backend
- Prediction probability handling
- User feedback collection
- Model retraining
- Structuring a machine learning project into separate training, inference, and preprocessing components

## Future Improvements

Possible improvements include:

- storing model artifacts using a dedicated model registry
- improving feedback validation
- adding automated tests
- containerizing the application with Docker
- adding experiment tracking
- deploying the API as a hosted service
