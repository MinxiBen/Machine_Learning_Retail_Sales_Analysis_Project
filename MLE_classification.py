import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import os
print(os.getcwd())
df= pd.read_excel("mle_month_sales202508.xlsx")

# 标签 y：销量是否超过阈值（>2 = 热销 1，否则 0）
df["hot"] = (df["店销数量"] >= 2).astype(int)

df.columns

target_date = pd.to_datetime("2025-08-31")
# 计算间隔天数（int）
df["days_interval"] = (target_date - df["上一次进货日"]).dt.days

X = df[["定价", "版别", "MC_R3 分类", "days_interval"]]
X = df[["定价", "版别", "MC_R3 分类", "days_interval"]].rename(
    columns={
        "定价": "price",
        "版别": "edition",
        "MC_R3 分类": "category",
        "days_interval": "days_interval"  # 已经是英文，可以不变
    }
)
Y = df["hot"]

# ========== 2. 特征工程 ==========
# 类别列 + 数值列
# OneHot 编码类别，数值特征标准化
categorical = ["edition", "category"]
numerical = ["price", "days_interval"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numerical)
    ]
)

# ========== 3. 模型训练 ==========
# 训练集 / 测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

# --- 3.1 SVM （加 class_weight 平衡类别）
svm_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", class_weight="balanced"))
])

svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)



print("SVM 分类结果：")
print(classification_report(y_test, y_pred_svm, target_names=["普通", "热销"]))


'''
SVM 分类结果：
              precision    recall  f1-score   support
          普通       0.98      0.91      0.94       216
          热销       0.38      0.80      0.51        15
    accuracy                           0.90       231
   macro avg       0.68      0.85      0.73       231
weighted avg       0.95      0.90      0.92       231
'''

# 误差分析：precision = (真热销/（真热销+判断为热销但实际普通)）
false_positives_index = X_test[(y_pred_svm == 1) & (y_test == 0)].index

print("假正例样本（预测为1但实际为0）:")
print(false_positives_index)

# 看哪些被误判为“热销”
df.loc[false_positives_index]