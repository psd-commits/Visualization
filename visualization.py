import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

sns.set(style="whitegrid")
df = pd.read_csv("C:/Users/GIA PHU/OneDrive/Tài liệu/NCKH/data/diabetic_data.csv")
print("Kích thước dữ liệu:", df.shape)   
df = df.replace("?", np.nan)
df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
df["age_mid"] = df["age"].str.extract(r"(\d+)-").astype(float) + 5
num_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "age_mid",
]
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(col)
plt.tight_layout()
plt.savefig("1_histograms.png", dpi=150)
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.countplot(data=df, x="gender", ax=axes[0])
axes[0].set_title("Giới tính")
sns.countplot(data=df, x="A1Cresult", order=["Norm", "<7", ">7", ">8"], ax=axes[1])
axes[1].set_title("Kết quả xét nghiệm A1C")
sns.countplot(data=df, x="readmitted", order=["NO", ">30", "<30"], ax=axes[2])
axes[2].set_title("Phân loại tái nhập viện (readmitted)")
plt.tight_layout()
plt.savefig("2_barcharts.png", dpi=150)
plt.show()
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="readmitted", y="time_in_hospital", order=["NO", ">30", "<30"])
plt.title("Thời gian nằm viện theo nhóm readmitted")
plt.tight_layout()
plt.savefig("3_boxplot_time.png", dpi=150)
plt.show()
plt.figure(figsize=(7, 5))
sns.violinplot(data=df, x="readmitted", y="num_lab_procedures", order=["NO", ">30", "<30"])
plt.title("Số xét nghiệm theo nhóm readmitted (Violin plot)")
plt.tight_layout()
plt.savefig("4_violin.png", dpi=150)
plt.show()
plt.figure(figsize=(6.5, 5.5))
sample = df.sample(3000, random_state=42)   
sns.scatterplot(data=sample, x="num_medications", y="time_in_hospital", alpha=0.25)
sns.regplot(data=df, x="num_medications", y="time_in_hospital", scatter=False, color="red")
plt.title("Số thuốc vs Thời gian nằm viện")
plt.tight_layout()
plt.savefig("5_scatter.png", dpi=150)
plt.show()
corr = df[num_cols + ["readmit_30"]].corr(numeric_only=True)
plt.figure(figsize=(8.5, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("6_correlation.png", dpi=150)
plt.show()
print("\nTương quan của từng biến với readmit_30 (sắp xếp giảm dần):")
print(corr["readmit_30"].drop("readmit_30").sort_values(ascending=False))
X = df[num_cols].fillna(df[num_cols].median())
y = df["readmit_30"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_s, y_train)
y_proba = clf.predict_proba(X_test_s)[:, 1]
y_pred = clf.predict(X_test_s)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="darkorange")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve — Dự đoán tái nhập viện <30 ngày")
plt.legend()
plt.savefig("7_roc_curve.png", dpi=150)
plt.show()
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Không <30d", "<30d"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("8_confusion_matrix.png", dpi=150)
plt.show()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_s)
plt.figure(figsize=(7, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train.values, alpha=0.3, palette="Set1")
plt.title(f"PCA 2D (giải thích {pca.explained_variance_ratio_.sum()*100:.1f}% biến thiên)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("9_pca_plot.png", dpi=150)
plt.show()
print("\n===== PHÁT HIỆN OUTLIER BẰNG IQR =====")
outlier_summary = []
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    outlier_summary.append({
        "bien": col, "Q1": q1, "Median": df[col].median(), "Q3": q3,
        "IQR": iqr, "nguong_duoi": low, "nguong_tren": high,
        "so_outlier": n_out, "ty_le_%": round(n_out / len(df) * 100, 2),
    })
outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))
outlier_df.to_csv("outlier_summary.csv", index=False)
grp = df.groupby("age")["time_in_hospital"]
stats = grp.agg(["mean", "std", "count"]).reset_index()
stats["se"] = stats["std"] / np.sqrt(stats["count"])        
stats["ci95"] = 1.96 * stats["se"]                             
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (err_col, ten) in zip(axes, [("std", "SD"), ("se", "SE"), ("ci95", "95% CI")]):
    ax.errorbar(stats["age"], stats["mean"], yerr=stats[err_col], fmt="o-", capsize=4, color="darkblue", ecolor="tomato")
    ax.set_title(f"Thời gian nằm viện trung bình ± {ten}")
    ax.set_xlabel("Nhóm tuổi")
    ax.tick_params(axis="x", rotation=45)
axes[0].set_ylabel("time_in_hospital (ngày)")
plt.tight_layout()
plt.savefig("10_error_bars.png", dpi=150)
plt.show()
trend = df.groupby("age").agg(ty_le_taonhapvien=("readmit_30", "mean"), so_thuoc_tb=("num_medications", "mean"),).reset_index()
trend["ty_le_taonhapvien"] *= 100
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(trend["age"], trend["ty_le_taonhapvien"], "o-", color="crimson", label="Tỉ lệ tái nhập viện <30 ngày (%)")
ax1.set_xlabel("Nhóm tuổi")
ax1.set_ylabel("Tỉ lệ tái nhập viện <30 ngày (%)", color="crimson")
ax1.tick_params(axis="x", rotation=45)
ax2 = ax1.twinx()
ax2.plot(trend["age"], trend["so_thuoc_tb"], "s--", color="steelblue", label="Số thuốc trung bình")
ax2.set_ylabel("Số thuốc trung bình", color="steelblue")
ax2.grid(False)
plt.title("Xu hướng theo nhóm tuổi")
fig.tight_layout()
plt.savefig("11_line_trend_by_age.png", dpi=150)
plt.show()
from scipy import stats as st
two = df[df["readmitted"].isin(["NO", "<30"])]
plt.figure(figsize=(7, 5))
sns.boxplot(data=two, x="readmitted", y="number_inpatient", order=["NO", "<30"])
sns.stripplot(data=two.sample(1500, random_state=1), x="readmitted", y="number_inpatient", order=["NO", "<30"], color="black", alpha=0.2, size=2)
plt.title("So sánh 2 nhóm: số lần nhập viện trước đó")
plt.tight_layout()
plt.savefig("12_two_group.png", dpi=150)
plt.show()
g1 = two[two["readmitted"] == "NO"]["number_inpatient"]
g2 = two[two["readmitted"] == "<30"]["number_inpatient"]
u, p = st.mannwhitneyu(g1, g2)
print(f"\nMann-Whitney U (2 nhóm): U={u:.1f}, p={p:.3e}")
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="readmitted", y="number_diagnoses", order=["NO", ">30", "<30"], inner="box")
plt.title("So sánh 3 nhóm: số chẩn đoán (number_diagnoses)")
plt.tight_layout()
plt.savefig("13_three_group.png", dpi=150)
plt.show()
groups = [df[df["readmitted"] == g]["number_diagnoses"] for g in ["NO", ">30", "<30"]]
h, p = st.kruskal(*groups)
print(f"Kruskal-Wallis (3 nhóm): H={h:.1f}, p={p:.3e}")
print("\nHOÀN TẤT TOÀN BỘ — các hình đã lưu từ 1_...png đến 13_...png")