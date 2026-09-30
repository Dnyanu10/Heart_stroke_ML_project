# ❤️ Heart Disease Prediction using Machine Learning Pipeline

A Machine Learning classification project that predicts the presence of **heart disease** using patient health-related attributes.

The project covers the complete ML workflow:

**Data Loading → EDA → Data Cleaning → Encoding → Train/Test Split → Feature Scaling → Model Training → Model Evaluation → Model Saving**

---

## 📌 Project Overview

The objective of this project is to build a machine learning pipeline that can classify whether a patient is likely to have heart disease based on available medical attributes.

Multiple classification algorithms are trained and compared using **Accuracy** and **F1 Score**.

The trained KNN model and preprocessing objects are also saved using `joblib` for potential future deployment.

> ⚠️ This project is for educational and machine-learning demonstration purposes only and should not be used as a medical diagnosis system.

---

## 🎯 Objectives

* Perform Exploratory Data Analysis (EDA)
* Identify missing, duplicate and invalid values
* Handle zero values in important numerical columns
* Analyze relationships between features and heart disease
* Convert categorical features into numerical features
* Split data into training and testing sets
* Apply feature scaling
* Train multiple classification algorithms
* Compare model performance
* Save the selected model and preprocessing artifacts

---

## 📊 Dataset

The project uses a heart disease dataset containing patient-related features.

### Important Features

| Feature        | Description                           |
| -------------- | ------------------------------------- |
| Age            | Age of the patient                    |
| Sex            | Gender                                |
| ChestPainType  | Type of chest pain                    |
| RestingBP      | Resting blood pressure                |
| Cholesterol    | Cholesterol level                     |
| FastingBS      | Fasting blood sugar                   |
| RestingECG     | Resting ECG result                    |
| MaxHR          | Maximum heart rate                    |
| ExerciseAngina | Exercise-induced angina               |
| Oldpeak        | ST depression                         |
| ST_Slope       | Slope of the peak exercise ST segment |
| HeartDisease   | Target variable                       |

---

## 🔎 Exploratory Data Analysis

The project performs several EDA operations:

* `head()`
* `shape`
* `info()`
* `describe()`
* Duplicate-value checking
* Null-value checking
* Target distribution analysis
* Histograms
* Count plots
* Box plots
* Violin plots
* Correlation heatmap

### Visual Analysis

The project analyzes relationships such as:

* Sex vs Heart Disease
* Chest Pain Type vs Heart Disease
* Fasting Blood Sugar vs Heart Disease
* Cholesterol vs Heart Disease
* Age vs Heart Disease
* Correlation between numerical features

---

## 🧹 Data Cleaning

The dataset contains zero values in some numerical columns where zero is not meaningful.

### Cholesterol

The mean cholesterol value excluding zero values is calculated and used to replace zero values.

```python
cholesterol_mean = df.loc[
    df['Cholesterol'] != 0,
    'Cholesterol'
].mean()

df['Cholesterol'] = df['Cholesterol'].replace(
    0,
    cholesterol_mean
)
```

### Resting Blood Pressure

Similarly, zero values in `RestingBP` are replaced using the mean of non-zero values.

```python
resting_bp_mean = df.loc[
    df['RestingBP'] != 0,
    'RestingBP'
].mean()

df['RestingBP'] = df['RestingBP'].replace(
    0,
    resting_bp_mean
)
```

---

## 🔄 Data Preprocessing

Categorical variables are converted into numerical variables using **One-Hot Encoding**.

```python
df_encoded = pd.get_dummies(
    df,
    drop_first=True
)

df_encoded = df_encoded.astype(int)
```

The target variable is separated from the input features:

```python
X = df_encoded.drop(
    'HeartDisease',
    axis=1
)

y = df_encoded['HeartDisease']
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42
)
```

### Split

* **80%** → Training data
* **20%** → Testing data
* `stratify=y` → Maintains the target-class distribution
* `random_state=42` → Reproducible results

---

## 📏 Feature Scaling

`StandardScaler` is used to standardize the features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

---

## 🤖 Machine Learning Models

The project compares five classification algorithms:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Naive Bayes**
4. **Decision Tree**
5. **Support Vector Machine (SVM with RBF Kernel)**

```python
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM (RBF Kernel)": SVC(probability=True)
}
```

---

## 📈 Model Evaluation

Each model is evaluated using:

### Accuracy

Measures the overall percentage of correctly classified observations.

```python
accuracy_score(y_test, y_pred)
```

### F1 Score

Provides a balance between precision and recall.

```python
f1_score(y_test, y_pred)
```

The results are stored in a comparison list:

```python
results.append({
    'Model': name,
    'Accuracy': round(acc, 4),
    'F1 Score': round(f1, 4)
})
```

---

## 💾 Model Saving

The KNN model, scaler and feature columns are saved using `joblib`.

```python
joblib.dump(
    models['KNN'],
    'KNN_heart.pkl'
)

joblib.dump(
    scaler,
    'scaler.pkl'
)

joblib.dump(
    X.columns.tolist(),
    'columns.pkl'
)
```

### Saved Files

| File            | Purpose                    |
| --------------- | -------------------------- |
| `KNN_heart.pkl` | Trained KNN model          |
| `scaler.pkl`    | Feature scaling object     |
| `columns.pkl`   | Feature-column information |

These files can be used later to build a prediction application or API.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

---

## 🔧 Machine Learning Workflow

```text
             Heart Disease Dataset
                      │
                      ▼
                Data Loading
                      │
                      ▼
                     EDA
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     Data Cleaning          Data Analysis
          │
          ▼
   One-Hot Encoding
          │
          ▼
   Train/Test Split
          │
          ▼
    Standard Scaling
          │
          ▼
   ┌──────┼─────────┬──────────┐
   ▼      ▼         ▼          ▼
 Logistic KNN    Naive Bayes Decision Tree
 Regression                   │
   │                          ▼
   └──────────► SVM ◄─────────┘
                   │
                   ▼
          Accuracy & F1 Score
                   │
                   ▼
             Model Saving
                   │
                   ▼
          KNN + Scaler + Columns
```

---

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── heart_prediction_pipeline.ipynb
├── KNN_heart.pkl
├── scaler.pkl
├── columns.pkl
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open:

```text
heart_prediction_pipeline.ipynb
```

### 4. Run the notebook

Execute the cells sequentially to reproduce the complete machine-learning workflow.

---

## 📌 Key Skills Demonstrated

* Exploratory Data Analysis
* Data Cleaning
* Handling invalid zero values
* Categorical Encoding
* Feature Engineering / Preprocessing
* Train-Test Split
* Stratified Sampling
* Feature Scaling
* Classification
* Model Comparison
* Accuracy & F1 Evaluation
* Model Serialization
* ML Pipeline Workflow

---

## 🔮 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Confusion matrix
* ROC-AUC evaluation
* Precision and recall comparison
* Feature importance analysis
* `Pipeline` and `ColumnTransformer`
* Prediction API using Flask/FastAPI
* Interactive Streamlit application
* Model monitoring
* Deployment to a cloud platform

---

## 👨‍💻 Author

**Dnyaneshwar Sonavane**

B.E. Information Technology

Interested in Data Analytics, Machine Learning and Data Science.

---

⭐ If you find this project useful, consider giving the repository a star!
