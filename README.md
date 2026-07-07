# 🩺 Disease Prediction using Machine Learning

Predict **Heart Disease, Diabetes, and Breast Cancer** using machine learning models trained on medical datasets with an easy-to-use Streamlit web application.

## 📌 Objective

Build a machine learning system that predicts the likelihood of multiple diseases from patient medical data, enabling faster and more accurate preliminary diagnosis.

## 🛠️ Approach

The project combines **data preprocessing**, **feature engineering**, and **supervised machine learning** to train classification models. Multiple algorithms are evaluated, and the best-performing model is deployed through a Streamlit interface for real-time predictions.

## ✨ Key Features

- **Multi-Disease Prediction**
  - Heart Disease
  - Diabetes
  - Breast Cancer
- **Data Preprocessing**
  - Missing value handling
  - Feature scaling
  - Data cleaning
- **Feature Engineering**
  - Feature selection and transformation
- **Model Training**
  - Train and compare multiple ML classifiers
- **Performance Evaluation**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- **Interactive Web App**
  - Streamlit-based interface for predictions
- **Model Persistence**
  - Save and reuse trained models

## 🧰 Tech Stack

- Python 3.9+
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Pickle

## 📂 Project Structure

```text
disease-prediction/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_input.json
├── models/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Heart_Disease.ipynb
│   ├── 03_Diabetes.ipynb
│   └── 04_Breast_Cancer.ipynb
├── results/
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── app.py
├── config.yaml
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/disease-prediction.git
cd disease-prediction

pip install -r requirements.txt
```

## ▶️ Usage

### 1. Prepare the dataset

Place the disease datasets inside the `data/raw/` directory.

### 2. Train the models

```bash
python src/train.py
```

This preprocesses the data, performs feature engineering, trains the models, and saves them in the `models/` directory.

### 3. Evaluate model performance

```bash
python src/evaluate.py
```

Displays performance metrics including:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### 4. Make predictions

```bash
python src/predict.py --input data/sample_input.json
```

Example output:

```text
Disease Prediction Results

Heart Disease : Negative
Diabetes      : Positive
Breast Cancer : Negative

Prediction Confidence:
Heart Disease : 96.8%
Diabetes      : 91.4%
Breast Cancer : 98.2%
```

### 5. Launch the Streamlit application

```bash
streamlit run app.py
```

Open the provided local URL in your browser, enter patient details, and receive disease predictions instantly.

## 📊 Results

| Disease | Best Model | Accuracy |
|----------|------------|----------|
| Heart Disease | XGBoost | ~90% |
| Diabetes | XGBoost | ~82% |
| Breast Cancer | SVM / XGBoost | ~97% |

The project demonstrates that machine learning can effectively assist in early disease prediction, providing fast, reliable, and scalable diagnostic support.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.

## 📄 License

This project is licensed under the MIT License.
