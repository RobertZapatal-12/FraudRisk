# FraudRisk

## About Me

Hi! I'm Yarel, an AI & Backend Developer focused on building practical AI systems that combine machine learning, backend engineering, and data security. 
I enjoy building projects where AI is not just a model, but an integral part of a complete software system—from data processing and model development to backend integration and deployment.
---

## About the Project

**FraudRisk** is a Machine Learning classification project focused on detecting potentially fraudulent credit card transactions.

The project takes raw transaction data, processes and cleans it, trains a classification model, and evaluates its performance on previously unseen data.

The main objective is not only to train a model, but to implement a complete and organized Machine Learning workflow where each stage has a clear responsibility.


Raw Dataset
     │
     ▼
Preprocessing
     │
     ▼
Processed Dataset
     │
     ▼
Model Training
     │
     ▼
Trained Model
     │
     ▼
Evaluation
     │
     ▼
Performance Metrics
```

---

## Project Structure

The project is organized into different components according to their responsibilities:

```text
Classifier/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   │
│   └── processed/
│       └── creditcard_clean.csv
│
├── models/
│   └── logistic_creditcard_model.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── training.py
│   │
│   └── __init__.py
│
├── .gitignore
├── pyproject.toml
└── uv.lock
```

### `data/`

Contains the datasets used throughout the project.

* `raw/` — Contains the original, unmodified dataset.
* `processed/` — Contains the cleaned and prepared dataset used for Machine Learning.

Keeping raw and processed data separate helps preserve the original dataset and makes the preprocessing workflow reproducible.

### `notebooks/`

Contains exploratory analysis.

`01_eda.ipynb` is used to investigate the dataset, understand its distributions, identify potential issues, and gain insights before building the model.

### `src/preprocessing/`

Responsible for preparing the data before it is passed to the Machine Learning model.

The main preprocessing logic is contained in:

```text
src/preprocessing/preprocess.py
```

This separation allows preprocessing to be reused independently from the training process.

### `src/training/`

Contains the logic responsible for training the Machine Learning model.

The training workflow is implemented in:

```text
src/training/training.py
```

The trained model is then saved in the `models/` directory.

### `models/`

Stores trained Machine Learning models.

Currently, the project contains:

```text
logistic_creditcard_model.pkl
```

This is a trained **Logistic Regression** model serialized for later use without requiring retraining.

### `src/evaluation/`

Contains the model evaluation logic.

The evaluation workflow is implemented in:

```text
src/evaluation/evaluate.py
```

This component is responsible for measuring how well the trained model performs on the data used for evaluation.

### `app/`

Contains the application entry point.

```text
app/main.py
```

This provides a separate layer for the application itself, keeping the Machine Learning pipeline independent from the application logic.

---

## Machine Learning Pipeline

The project follows a modular Machine Learning workflow:

### 1. Exploratory Data Analysis

The dataset is first explored using the EDA notebook to understand:

* Dataset structure
* Feature distributions
* Missing values
* Class distribution
* Potential anomalies
* Relationships between variables

### 2. Data Preprocessing

The raw dataset is cleaned and transformed into a format suitable for Machine Learning.

The resulting dataset is stored separately in:

```text
data/processed/creditcard_clean.csv
```

### 3. Model Training

The processed data is used to train a **Logistic Regression** classification model.

The training logic is separated from preprocessing so that each stage can be developed and tested independently.

### 4. Model Evaluation

After training, the model is evaluated using classification metrics to determine how effectively it distinguishes between legitimate and fraudulent transactions.

---

## Model

The current implementation uses **Logistic Regression** as the classification algorithm.

Logistic Regression is a good baseline for binary classification because it is relatively simple, interpretable, and provides a strong starting point for comparing more complex models in future iterations.

The trained model is stored as:

```text
models/logistic_creditcard_model.pkl
```

---

## Technologies

The project is built primarily with Python and the following technologies:

* **Python**
* **Pandas** — Data manipulation and preprocessing
* **Scikit-learn** — Machine Learning
* **Jupyter Notebook** — Exploratory Data Analysis
* **Git** — Version control
* **uv** — Python dependency and environment management

---

## Project Goals

The main goal of FraudRisk is to practice the complete Machine Learning development workflow rather than treating model training as an isolated task.

Through this project, I am developing experience with:

* Exploratory Data Analysis
* Data preprocessing
* Binary classification
* Logistic Regression
* Model serialization
* Model evaluation
* Modular Python project architecture
* Separation of responsibilities
* Reproducible Machine Learning workflows
* Git-based project management

---

## Future Improvements

The project can be extended in several directions:

* Compare Logistic Regression with other classification algorithms
* Address class imbalance more extensively
* Perform feature engineering
* Add cross-validation
* Implement hyperparameter optimization
* Add additional evaluation metrics
* Build a prediction API
* Add automated tests
* Containerize the application with Docker
* Implement a complete inference pipeline

---

## Author

**Yarel Zapata**

Artificial Intelligence Student | AI & Data Developer

Interested in **Machine Learning, Artificial Intelligence, Data Engineering, and Software Development**.
