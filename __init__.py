{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 📊 Exploratory Data Analysis — Disease Prediction\n", "Explore the three datasets before modelling."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, '../src')\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from data_loader import load_dataset, download_all\n",
    "\n",
    "sns.set_theme(style='whitegrid', palette='muted')\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download datasets if needed\n",
    "download_all()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Heart Disease Dataset"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_h, y_h = load_dataset('heart')\n",
    "df_heart = X_h.copy()\n",
    "df_heart['target'] = y_h\n",
    "print(df_heart.shape)\n",
    "df_heart.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Class distribution\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "\n",
    "for ax, (name, ds_name) in zip(axes, [('Heart Disease', 'heart'),\n",
    "                                        ('Diabetes', 'diabetes'),\n",
    "                                        ('Breast Cancer', 'breast_cancer')]):\n",
    "    X, y = load_dataset(ds_name)\n",
    "    y.value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])\n",
    "    ax.set_title(f'{name} — Class Distribution')\n",
    "    ax.set_xlabel('Class')\n",
    "    ax.set_ylabel('Count')\n",
    "    ax.tick_params(axis='x', rotation=0)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../results/class_distributions.png', dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation heatmap — Heart Disease\n",
    "plt.figure(figsize=(12, 8))\n",
    "sns.heatmap(df_heart.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)\n",
    "plt.title('Correlation Heatmap — Heart Disease')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../results/heart_correlation.png', dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Missing value analysis\n",
    "print('Heart Disease missing values:')\n",
    "print(df_heart.isnull().sum())\n",
    "df_heart.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Diabetes Dataset"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_d, y_d = load_dataset('diabetes')\n",
    "df_diab = X_d.copy()\n",
    "df_diab['Outcome'] = y_d\n",
    "\n",
    "# Zero values in medical features are physiologically impossible — treat as missing\n",
    "zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']\n",
    "for col in zero_invalid_cols:\n",
    "    print(f'{col}: {(df_diab[col] == 0).sum()} zero values')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature distributions by class\n",
    "features = ['Glucose', 'BMI', 'Age', 'Insulin']\n",
    "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
    "for ax, feat in zip(axes.flat, features):\n",
    "    for label, color in [(0, 'steelblue'), (1, 'tomato')]:\n",
    "        df_diab[df_diab['Outcome'] == label][feat].hist(\n",
    "            ax=ax, alpha=0.6, color=color, bins=20, label=str(label))\n",
    "    ax.set_title(feat)\n",
    "    ax.legend(['No Diabetes', 'Diabetes'])\n",
    "plt.suptitle('Diabetes — Feature Distributions by Class')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../results/diabetes_distributions.png', dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Breast Cancer Dataset"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_bc, y_bc = load_dataset('breast_cancer')\n",
    "df_bc = X_bc.copy()\n",
    "df_bc['diagnosis'] = y_bc\n",
    "\n",
    "# Most correlated features with target\n",
    "corr = df_bc.corr()['diagnosis'].drop('diagnosis').abs().sort_values(ascending=False)\n",
    "print('Top 10 features correlated with diagnosis:')\n",
    "print(corr.head(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pairplot of top features\n",
    "top_feats = corr.head(4).index.tolist() + ['diagnosis']\n",
    "sns.pairplot(df_bc[top_feats], hue='diagnosis', palette={0: 'steelblue', 1: 'tomato'})\n",
    "plt.suptitle('Breast Cancer — Top Feature Pairplot', y=1.02)\n",
    "plt.savefig('../results/breast_cancer_pairplot.png', dpi=100)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
