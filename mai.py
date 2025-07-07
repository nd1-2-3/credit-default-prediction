import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier


 
column_names = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "EmploymentDuration",
    "InstallmentRate", "PersonalStatusSex", "OtherDebtors", "PresentResidence", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "NumberExistingCredits", "Job", "NumberPeopleLiable",
    "Telephone", "ForeignWorker", "Target"
]

raw_df = pd.read_csv("//Users//jaimeen//Downloads//statlog+german+credit+data//german.data", sep=' ', header=None, names=column_names)

numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove("Target")
cat_cols = raw_df.select_dtypes('object').columns.tolist()

train_df, temp_df = train_test_split(raw_df, test_size=0.3, random_state=42 )
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42 )

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Numeric columns 
train_df[numeric_cols] = num_imputer.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = num_imputer.transform(test_df[numeric_cols])
val_df[numeric_cols] = num_imputer.transform(val_df[numeric_cols])

# Categorical columns
train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])
val_df[cat_cols] = cat_imputer.transform(val_df[cat_cols])


scaler = MinMaxScaler()

# Fit on train, transform on all
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])


encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoder.fit(train_df[cat_cols])

train_temp_df = pd.DataFrame(encoder.transform(train_df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=train_df.index)
test_temp_df = pd.DataFrame(encoder.transform(test_df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=test_df.index)
val_temp_df = pd.DataFrame(encoder.transform(val_df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=val_df.index)

# 4️⃣ Combine
train_final = pd.concat([train_df.drop(columns=cat_cols), train_temp_df], axis=1)
test_final = pd.concat([test_df.drop(columns=cat_cols), test_temp_df], axis=1)
val_final = pd.concat([val_df.drop(columns=cat_cols), val_temp_df], axis=1)

train_inputs = train_final.drop(columns=["Target"])
train_target = train_final["Target"].map({1: 1, 2: 0})

test_inputs = test_final.drop(columns=["Target"])
test_target = test_final["Target"].map({1: 1, 2: 0})

val_inputs = val_final.drop(columns=["Target"])
val_target = val_final["Target"].map({1: 1, 2: 0})

log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg.fit(train_inputs, train_target)

test_preds = log_reg.predict(test_inputs)
test_probs = log_reg.predict_proba(test_inputs)[:, 1]


conf_matrix = confusion_matrix(test_target, test_preds)

print("Confusion Matrix:\n", conf_matrix)


plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
#plt.show()


print("\nClassification Report:\n", classification_report(test_target, test_preds))

print("Accuracy Score:", accuracy_score(test_target, test_preds))


#coef_series = pd.Series(log_reg.coef_[0], index=train_inputs.columns)
#print(coef_series.abs().sort_values(ascending=False).head(30))

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(train_inputs, train_target)

y_test_preds_xgb = xgb_model.predict(test_inputs)
y_test_probs_xgb = xgb_model.predict_proba(test_inputs)[:, 1]


conf_matrix_xgb = confusion_matrix(test_target, y_test_preds_xgb)
print("Confusion Matrix:\n", conf_matrix_xgb)


plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("XGBoost Confusion Matrix")
#plt.show()


print("\nClassification Report:\n", classification_report(test_target, y_test_preds_xgb))


print("Accuracy Score:", accuracy_score(test_target, y_test_preds_xgb))


coef_series = pd.Series(log_reg.coef_[0], index=train_inputs.columns)
coef_series_sorted = coef_series.abs().sort_values(ascending=False).head(15)


importance_series = pd.Series(xgb_model.feature_importances_, index=train_inputs.columns)
importance_sorted = importance_series.sort_values(ascending=False).head(15)

combined_df = pd.DataFrame({
    "Logistic Regression": coef_series_sorted,
    "XGBoost": importance_sorted
}).fillna(0) 


combined_df = combined_df.sort_values(by="Logistic Regression", ascending=True)


plt.figure(figsize=(10, 8))
combined_df.plot(kind="barh", figsize=(10, 8), color=["skyblue", "orange"])
plt.xlabel("Importance / Coefficient Magnitude")
plt.title("Feature Importances: Logistic Regression vs XGBoost")
plt.grid(axis='x')
plt.tight_layout()
plt.show()

