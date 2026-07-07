{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 🩺 Diabetes Prediction\n", "Train and evaluate all 4 classifiers on the Pima Indians Diabetes dataset."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, '../src')\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.metrics import roc_curve, roc_auc_score\n",
    "\n",
    "from data_loader import load_dataset\n",
    "from feature_engineering import preprocess, feature_importance_summary\n",
    "from evaluate import evaluate_model\n",
    "\n",
    "sns.set_theme(style='whitegrid')\n",
    "%matplotlib inline\n",
    "\n",
    "DATASET = 'diabetes'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X, y = load_dataset(DATASET)\n",
    "\n",
    "# Replace physiologically impossible zero values with NaN (imputed later)\n",
    "zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']\n",
    "X[zero_invalid] = X[zero_invalid].replace(0, np.nan)\n",
    "\n",
    "X_train, X_test, y_train, y_test, pipeline = preprocess(X, y, DATASET)\n",
    "print(f'Train: {X_train.shape} | Test: {X_test.shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),\n",
    "    'svm': SVC(kernel='rbf', probability=True, random_state=42),\n",
    "    'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),\n",
    "    'xgboost': XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', verbosity=0),\n",
    "}\n",
    "\n",
    "results = {}\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    metrics = evaluate_model(model, X_test, y_test, name, DATASET)\n",
    "    results[name] = metrics\n",
    "    print(f'{name}: Acc={metrics[\"accuracy\"]:.4f} | F1={metrics[\"f1_score\"]:.4f} | AUC={metrics[\"roc_auc\"]:.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature importances\n",
    "rf = models['random_forest']\n",
    "feat_df = feature_importance_summary(rf, X.columns.tolist())\n",
    "plt.figure(figsize=(8, 4))\n",
    "sns.barplot(data=feat_df, x='importance', y='feature', palette='Greens_d')\n",
    "plt.title('Feature Importance — Random Forest (Diabetes)')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../results/diabetes_feature_importance.png', dpi=150)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.9.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
