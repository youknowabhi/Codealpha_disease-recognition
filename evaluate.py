{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# ❤️ Heart Disease Prediction\n", "Train and evaluate all 4 classifiers on the Cleveland Heart Disease dataset."]
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
    "from sklearn.metrics import classification_report, roc_curve, roc_auc_score\n",
    "\n",
    "from data_loader import load_dataset\n",
    "from feature_engineering import preprocess, feature_importance_summary\n",
    "from evaluate import evaluate_model, plot_roc_curves, plot_model_comparison\n",
    "\n",
    "sns.set_theme(style='whitegrid')\n",
    "%matplotlib inline\n",
    "\n",
    "DATASET = 'heart'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X, y = load_dataset(DATASET)\n",
    "X_train, X_test, y_train, y_test, pipeline = preprocess(X, y, DATASET)\n",
    "print(f'Train: {X_train.shape} | Test: {X_test.shape}')\n",
    "print(f'Class distribution (train): {pd.Series(y_train).value_counts().to_dict()}')"
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
    "# ROC curves\n",
    "plt.figure(figsize=(8, 6))\n",
    "colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']\n",
    "for (name, model), color in zip(models.items(), colors):\n",
    "    y_prob = model.predict_proba(X_test)[:, 1]\n",
    "    fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
    "    auc = roc_auc_score(y_test, y_prob)\n",
    "    plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.2f})')\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('ROC Curves — Heart Disease')\n",
    "plt.legend(loc='lower right')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../results/heart_roc_curves.png', dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature importance — Random Forest\n",
    "rf = models['random_forest']\n",
    "feat_df = feature_importance_summary(rf, X.columns.tolist())\n",
    "plt.figure(figsize=(8, 5))\n",
    "sns.barplot(data=feat_df, x='importance', y='feature', palette='Blues_d')\n",
    "plt.title('Feature Importance — Random Forest (Heart Disease)')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../results/heart_feature_importance.png', dpi=150)\n",
    "plt.show()\n",
    "feat_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Detailed classification report for best model (XGBoost)\n",
    "print(results['xgboost']['classification_report'])"
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
