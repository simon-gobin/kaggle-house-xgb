
# 🏠 Kaggle House Prices – XGBoost Regression Pipeline

End-to-end machine learning pipeline for the Kaggle **House Prices: Advanced Regression Techniques** competition using:

- Feature preprocessing with Scikit-learn
- XGBoost regression (CPU / GPU compatible)
- Parallel hyperparameter tuning (GridSearchCV)
- Automated Kaggle download & submission
- Google Colab / Local support

---

## 🚀 Features

✔ Automatic Kaggle dataset download  
✔ Robust preprocessing (imputation + encoding)  
✔ Feature filtering (low variance removal)  
✔ Parallel hyperparameter search (CPU)  
✔ XGBoost training (CPU / GPU)  
✔ Automatic submission to Kaggle  
✔ Experiment logging  
✔ Reproducible pipeline  

---

## 📂 Project Structure

```
kaggle-house-xgb/
│
├── main.py
├── requirements.txt
├── README.md
├── grid_results.csv
└── submission.csv
```

---

## 📊 Competition

House Prices – Advanced Regression Techniques  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Evaluation metric: RMSE on log(SalePrice)

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/simon-gobin/kaggle-house-xgb.git
cd kaggle-house-xgb
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Kaggle API Setup

1. Go to https://www.kaggle.com/account  
2. Click "Create New API Token"  
3. Download kaggle.json

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Test:

```bash
kaggle config view
```

---

## ▶️ Run the Pipeline

```bash
python main.py
```

The script will:
- Download data
- Preprocess features
- Run GridSearchCV
- Train best model
- Generate submission
- Submit to Kaggle

---

## ☁️ Google Colab Usage

```python
!git clone https://github.com/simon-gobin/kaggle-house-xgb.git
%cd kaggle-house-xgb
!pip install -r requirements.txt
```

Upload token:

```python
from google.colab import files
files.upload()
```

```bash
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Run:

```bash
!python main.py
```

---

## 🧠 Machine Learning Pipeline

- Median imputation (numerical)
- Most frequent + ordinal encoding (categorical)
- Variance filtering
- XGBoost regressor
- GridSearchCV (5-fold)

---

## 📈 Tuned Hyperparameters

- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight
- reg_lambda
- reg_alpha

---

## 📁 Outputs

| File | Description |
|------|-------------|
| submission.csv | Kaggle submission |
| grid_results.csv | Grid search results |

---

## 👨‍💻 Author

Simon Gobin  
GitHub: https://github.com/simon-gobin

---

## 📜 License

MIT License

Free to use for learning and portfolio purposes.

---

Happy modeling 🚀
