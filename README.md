# credit-default-prediction

# Credit Default Prediction: Logistic Regression vs XGBoost

This project focuses on predicting credit risk defaults using the German Credit dataset. The main goal is to build an end-to-end, interpretable, and practical pipeline to help financial institutions identify high-risk applicants and improve lending decisions.

---

## 💼 Project Overview

- **Business context:** In credit risk modeling, accurately identifying potential defaulters is critical for reducing financial losses and improving portfolio health.
- **Approach:** Compared a traditional interpretable model (Logistic Regression) with a powerful ensemble model (XGBoost) to understand trade-offs between interpretability and predictive power.

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Handled missing values using median (for numerical) and most frequent (for categorical) imputation strategies.
   - Scaled numerical features to the [0, 1] range using MinMaxScaler.
   - Applied one-hot encoding to categorical variables, avoiding data leakage by fitting on training data only.

2. **Modeling**
   - **Logistic Regression:** Widely used in credit scoring for its interpretability and linear decision boundary.
   - **XGBoost Classifier:** Captures complex, non-linear interactions and often used for boosting predictive performance.

3. **Evaluation**
   - Compared models using accuracy, confusion matrices, and detailed classification reports.
   - Visualized feature importances side by side to understand drivers behind predictions.

---

## 🟢 Results

| Model              | Test Accuracy |
|-------------------|---------------|
| Logistic Regression | **80%**       |
| XGBoost            | **74.66%**    |

- Logistic Regression achieved higher accuracy and remains easier to explain to risk committees and regulators.
- XGBoost, while slightly lower in accuracy here, captures non-linearities and provides complementary insights into feature relationships.


## 💡 Key Learnings

- Importance of thorough preprocessing and avoiding data leakage
- Trade-offs between model interpretability and predictive power
- Techniques to compare and interpret feature importances to support business decision-making

---

## 🚀 Tech Stack

- Python
- scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 📄 Files

- `main.py`: Core pipeline including preprocessing, model training, evaluation, and visualizations.
- `plots/`: Contains generated plots (feature importance and confusion matrices).

---

## 🤝 Let's Connect

If you'd like to discuss this project, quantitative finance, or data science applications in risk modeling, feel free to connect!

[LinkedIn](https://www.linkedin.com/nand-davda) | [Email](mailto:your-email@nand.davda18@gmail.com)

---

### ⭐ Feel free to fork, open issues, or suggest improvements!
