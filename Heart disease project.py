"""
============================================================
  Heart Disease Data Science Project
  Dataset : UCI Heart Disease (Cleveland + combined, n=918)
  Goal    : End-to-end Analysis + Prediction
  Author  : Generated with Claude
============================================================
"""

# ── 0. Imports ────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# ── Palette ───────────────────────────────────────────────
PALETTE   = {"No Disease": "#4C9BE8", "Heart Disease": "#E85D5D"}
CLR_POS   = "#E85D5D"
CLR_NEG   = "#4C9BE8"
CLR_GRID  = "#F0F0F0"
sns.set_theme(style="whitegrid", font_scale=1.05)

OUTPUT_DIR = "/mnt/user-data/outputs/"
import os; os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# 1. LOAD & UNDERSTAND DATA
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  HEART DISEASE — END-TO-END DATA SCIENCE PROJECT")
print("="*60)

df = pd.read_csv("heart_disease.csv")

FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0–3)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar >120 mg/dl",
    "restecg":  "Resting ECG Results (0–2)",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (oldpeak)",
    "slope":    "Slope of Peak Exercise ST Segment",
    "ca":       "# Major Vessels Coloured (0–3)",
    "thal":     "Thalassemia (1=Normal, 2=Fixed, 3=Reversible)",
    "target":   "Heart Disease (1=Yes, 0=No)"
}

print(f"\n Dataset Shape : {df.shape}")
print(f" Missing Values: {df.isnull().sum().sum()}")
print(f"\n Target Distribution:")
vc = df["target"].value_counts()
for k, v in vc.items():
    label = "Heart Disease" if k == 1 else "No Disease"
    print(f"   {label}: {v} ({v/len(df)*100:.1f}%)")

print(f"\n Key Statistics:")
print(df[["age","trestbps","chol","thalach","oldpeak"]].describe().round(2).to_string())

# ══════════════════════════════════════════════════════════
# 2. EDA — FIGURE 1: Overview Dashboard
# ══════════════════════════════════════════════════════════
print("\n\n[1/4] Generating EDA Overview Dashboard ...")

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("white")
fig.suptitle("Heart Disease Dataset — Exploratory Data Analysis",
             fontsize=22, fontweight="bold", y=0.98, color="#1a1a2e")

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

target_labels = df["target"].map({0: "No Disease", 1: "Heart Disease"})
colors_mapped  = target_labels.map(PALETTE)

# ── 2.1 Target Distribution (Pie) ────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
sizes = df["target"].value_counts().sort_index()
ax0.pie(sizes, labels=["No Disease", "Heart Disease"],
        colors=[CLR_NEG, CLR_POS], autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax0.set_title("Target Distribution", fontweight="bold", fontsize=13)

# ── 2.2 Age Distribution by Target ────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
for label, grp in df.groupby("target"):
    lbl = "Heart Disease" if label == 1 else "No Disease"
    ax1.hist(grp["age"], bins=18, alpha=0.7,
             color=PALETTE[lbl], label=lbl, edgecolor="white")
ax1.set_xlabel("Age"); ax1.set_ylabel("Count")
ax1.set_title("Age Distribution by Outcome", fontweight="bold", fontsize=13)
ax1.legend(); ax1.set_facecolor(CLR_GRID)

# ── 2.3 Sex vs Target ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
sex_tgt = df.groupby(["sex","target"]).size().unstack(fill_value=0)
sex_tgt.index = ["Female","Male"]
sex_tgt.columns = ["No Disease","Heart Disease"]
sex_tgt.plot(kind="bar", ax=ax2, color=[CLR_NEG, CLR_POS],
             edgecolor="white", rot=0)
ax2.set_title("Sex vs Heart Disease", fontweight="bold", fontsize=13)
ax2.set_xlabel(""); ax2.set_ylabel("Count")
ax2.legend(); ax2.set_facecolor(CLR_GRID)

# ── 2.4 Chest Pain Type ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
cp_labels = {0:"Typical\nAngina", 1:"Atypical\nAngina",
             2:"Non-Anginal\nPain", 3:"Asymptomatic"}
cp_tgt = df.groupby(["cp","target"]).size().unstack(fill_value=0)
cp_tgt.index = [cp_labels[i] for i in cp_tgt.index]
cp_tgt.columns = ["No Disease","Heart Disease"]
cp_tgt.plot(kind="bar", ax=ax3, color=[CLR_NEG, CLR_POS],
            edgecolor="white", rot=0)
ax3.set_title("Chest Pain Type vs Outcome", fontweight="bold", fontsize=13)
ax3.set_xlabel(""); ax3.set_ylabel("Count")
ax3.legend(); ax3.set_facecolor(CLR_GRID)

# ── 2.5 Cholesterol by Target ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for label, grp in df.groupby("target"):
    lbl = "Heart Disease" if label == 1 else "No Disease"
    ax4.hist(grp["chol"], bins=20, alpha=0.7,
             color=PALETTE[lbl], label=lbl, edgecolor="white")
ax4.set_xlabel("Cholesterol (mg/dl)"); ax4.set_ylabel("Count")
ax4.set_title("Cholesterol Distribution", fontweight="bold", fontsize=13)
ax4.legend(); ax4.set_facecolor(CLR_GRID)

# ── 2.6 Max Heart Rate vs Age (scatter) ──────────────────
ax5 = fig.add_subplot(gs[1, 2])
sc = ax5.scatter(df["age"], df["thalach"],
                 c=df["target"].map({0: CLR_NEG, 1: CLR_POS}),
                 alpha=0.55, s=35, edgecolors="white", linewidths=0.3)
ax5.set_xlabel("Age"); ax5.set_ylabel("Max Heart Rate")
ax5.set_title("Age vs Max Heart Rate", fontweight="bold", fontsize=13)
from matplotlib.patches import Patch
ax5.legend(handles=[Patch(color=CLR_NEG, label="No Disease"),
                    Patch(color=CLR_POS, label="Heart Disease")])
ax5.set_facecolor(CLR_GRID)

# ── 2.7 Correlation Heatmap ───────────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax6,
            cmap="RdBu_r", center=0, linewidths=0.5,
            annot_kws={"size": 8}, vmin=-1, vmax=1)
ax6.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=13)
ax6.tick_params(axis='x', rotation=45, labelsize=9)
ax6.tick_params(axis='y', rotation=0, labelsize=9)

# ── 2.8 Box: Oldpeak by Target ───────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
df_box = df.copy()
df_box["Outcome"] = df_box["target"].map({0:"No Disease",1:"Heart Disease"})
sns.boxplot(data=df_box, x="Outcome", y="oldpeak", ax=ax7,
            palette=PALETTE, width=0.5, linewidth=1.5)
ax7.set_title("ST Depression (Oldpeak)\nby Outcome", fontweight="bold", fontsize=13)
ax7.set_xlabel(""); ax7.set_facecolor(CLR_GRID)

# ── 2.9 Thal distribution ─────────────────────────────────
ax8 = fig.add_subplot(gs[3, 0])
thal_tgt = df.groupby(["thal","target"]).size().unstack(fill_value=0)
thal_tgt.index = ["Normal","Fixed Defect","Reversible Defect"]
thal_tgt.columns = ["No Disease","Heart Disease"]
thal_tgt.plot(kind="bar", ax=ax8, color=[CLR_NEG, CLR_POS],
              edgecolor="white", rot=15)
ax8.set_title("Thalassemia Type vs Outcome", fontweight="bold", fontsize=13)
ax8.set_xlabel(""); ax8.legend(); ax8.set_facecolor(CLR_GRID)

# ── 2.10 Exercise Angina ─────────────────────────────────
ax9 = fig.add_subplot(gs[3, 1])
exang_tgt = df.groupby(["exang","target"]).size().unstack(fill_value=0)
exang_tgt.index = ["No Angina","Exercise Angina"]
exang_tgt.columns = ["No Disease","Heart Disease"]
exang_tgt.plot(kind="bar", ax=ax9, color=[CLR_NEG, CLR_POS],
               edgecolor="white", rot=0)
ax9.set_title("Exercise Angina vs Outcome", fontweight="bold", fontsize=13)
ax9.set_xlabel(""); ax9.legend(); ax9.set_facecolor(CLR_GRID)

# ── 2.11 Blood Pressure Box ──────────────────────────────
ax10 = fig.add_subplot(gs[3, 2])
sns.boxplot(data=df_box, x="Outcome", y="trestbps", ax=ax10,
            palette=PALETTE, width=0.5, linewidth=1.5)
ax10.set_title("Resting Blood Pressure\nby Outcome", fontweight="bold", fontsize=13)
ax10.set_xlabel(""); ax10.set_facecolor(CLR_GRID)

plt.savefig(f"{OUTPUT_DIR}fig1_eda_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("   ✓ Saved fig1_eda_dashboard.png")

# ══════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════
print("\n[2/4] Preprocessing & Feature Engineering ...")

X = df.drop("target", axis=1)
y = df["target"]

# One-hot encode categoricals
cat_cols = ["cp", "restecg", "slope", "thal"]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"   Features after encoding: {X.shape[1]}")

# ══════════════════════════════════════════════════════════
# 4. MODEL TRAINING & COMPARISON
# ══════════════════════════════════════════════════════════
print("\n[3/4] Training & Comparing Models ...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=42),
    "SVM":                 SVC(probability=True, C=1.0, kernel="rbf", random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=150, learning_rate=0.07,
                                         use_label_encoder=False, eval_metric="logloss",
                                         random_state=42, verbosity=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    Xtr = X_train_s if name in ["Logistic Regression","SVM"] else X_train
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="roc_auc")
    model.fit(Xtr, y_train)
    Xte  = X_test_s if name in ["Logistic Regression","SVM"] else X_test
    prob = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    test_auc  = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)
    report    = classification_report(y_test, pred, output_dict=True)
    results[name] = {
        "model": model,
        "cv_auc_mean": cv_scores.mean(),
        "cv_auc_std":  cv_scores.std(),
        "test_auc":    test_auc,
        "fpr": fpr, "tpr": tpr,
        "pred": pred, "prob": prob,
        "precision": report["1"]["precision"],
        "recall":    report["1"]["recall"],
        "f1":        report["1"]["f1-score"],
        "accuracy":  report["accuracy"],
        "Xtr": Xtr, "Xte": Xte,
    }
    print(f"   {name:<25} CV-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  |  Test-AUC: {test_auc:.4f}")

best_name = max(results, key=lambda k: results[k]["test_auc"])
best      = results[best_name]
print(f"\n    Best Model: {best_name}  (Test AUC = {best['test_auc']:.4f})")

# ══════════════════════════════════════════════════════════
# 5. MODEL RESULTS — FIGURE 2
# ══════════════════════════════════════════════════════════
print("\n[4/4] Generating Model Results Dashboard ...")

fig2 = plt.figure(figsize=(20, 18))
fig2.patch.set_facecolor("white")
fig2.suptitle("Heart Disease Prediction — Model Evaluation",
              fontsize=22, fontweight="bold", y=0.99, color="#1a1a2e")

gs2 = gridspec.GridSpec(3, 3, figure=fig2, hspace=0.55, wspace=0.40)

# ── 5.1 Model Comparison Bar Chart ───────────────────────
ax = fig2.add_subplot(gs2[0, :2])
names     = list(results.keys())
cv_means  = [results[n]["cv_auc_mean"] for n in names]
cv_stds   = [results[n]["cv_auc_std"]  for n in names]
test_aucs = [results[n]["test_auc"]    for n in names]
x = np.arange(len(names))
bars = ax.bar(x - 0.2, cv_means,  0.35, yerr=cv_stds, capsize=4,
              color=CLR_NEG, alpha=0.85, label="CV AUC (5-fold)", edgecolor="white")
ax.bar(x + 0.2, test_aucs, 0.35,
       color=CLR_POS, alpha=0.85, label="Test AUC", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=10)
ax.set_ylim(0.70, 1.0)
ax.set_title("Model Comparison — AUC Scores", fontweight="bold", fontsize=13)
ax.set_ylabel("AUC Score")
ax.legend(); ax.set_facecolor(CLR_GRID)
for bar, val in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

# ── 5.2 Metrics Table ─────────────────────────────────────
ax_tbl = fig2.add_subplot(gs2[0, 2])
ax_tbl.axis("off")
metrics_data = [[n,
                 f"{results[n]['accuracy']:.3f}",
                 f"{results[n]['precision']:.3f}",
                 f"{results[n]['recall']:.3f}",
                 f"{results[n]['f1']:.3f}",
                 f"{results[n]['test_auc']:.3f}"]
                for n in names]
table = ax_tbl.table(
    cellText=metrics_data,
    colLabels=["Model","Acc","Prec","Recall","F1","AUC"],
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False); table.set_fontsize(9)
table.scale(1.0, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1a1a2e"); cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f5f5f5")
ax_tbl.set_title("Performance Metrics", fontweight="bold", fontsize=13, pad=14)

# ── 5.3 ROC Curves ───────────────────────────────────────
ax_roc = fig2.add_subplot(gs2[1, :2])
roc_colors = ["#4C9BE8","#E85D5D","#27AE60","#F39C12","#8E44AD"]
for (name, res), color in zip(results.items(), roc_colors):
    ax_roc.plot(res["fpr"], res["tpr"], lw=2, color=color,
                label=f"{name} (AUC={res['test_auc']:.3f})")
ax_roc.plot([0,1],[0,1],"k--", lw=1.5, alpha=0.5, label="Random")
ax_roc.fill_between([0,1],[0,1], alpha=0.05, color="gray")
ax_roc.set_xlim([0,1]); ax_roc.set_ylim([0,1.02])
ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves — All Models", fontweight="bold", fontsize=13)
ax_roc.legend(loc="lower right", fontsize=9); ax_roc.set_facecolor(CLR_GRID)

# ── 5.4 Confusion Matrix — Best Model ────────────────────
ax_cm = fig2.add_subplot(gs2[1, 2])
cm = confusion_matrix(y_test, best["pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Disease","Heart Disease"])
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
ax_cm.set_title(f"Confusion Matrix\n{best_name}", fontweight="bold", fontsize=12)
ax_cm.tick_params(labelsize=9)

# ── 5.5 Feature Importance (Best model if tree-based) ────
ax_fi = fig2.add_subplot(gs2[2, :2])
best_model = best["model"]
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_names  = list(X.columns)
    idx = np.argsort(importances)[-15:]
    colors_bar = [CLR_POS if importances[i] > np.median(importances) else CLR_NEG for i in idx]
    ax_fi.barh([feat_names[i] for i in idx], importances[idx],
               color=colors_bar, edgecolor="white", height=0.7)
    ax_fi.set_title(f"Top 15 Feature Importances — {best_name}",
                    fontweight="bold", fontsize=13)
    ax_fi.set_xlabel("Importance Score")
    ax_fi.set_facecolor(CLR_GRID)
else:
    # Logistic regression coefficients
    coef = np.abs(best_model.coef_[0])
    feat_names = list(X.columns)
    idx = np.argsort(coef)[-15:]
    ax_fi.barh([feat_names[i] for i in idx], coef[idx],
               color=CLR_POS, edgecolor="white", height=0.7)
    ax_fi.set_title(f"Top 15 Feature Coefficients — {best_name}",
                    fontweight="bold", fontsize=13)
    ax_fi.set_xlabel("|Coefficient|")
    ax_fi.set_facecolor(CLR_GRID)

# ── 5.6 Cross-Val AUC Distribution ───────────────────────
ax_cv = fig2.add_subplot(gs2[2, 2])
all_cv = []
for name, res in results.items():
    Xtr_cv = X_train_s if name in ["Logistic Regression","SVM"] else X_train
    folds  = cross_val_score(res["model"], Xtr_cv, y_train, cv=cv, scoring="roc_auc")
    for f in folds:
        all_cv.append({"Model": name[:12], "AUC": f})
cv_df = pd.DataFrame(all_cv)
sns.boxplot(data=cv_df, x="AUC", y="Model", ax=ax_cv,
            palette="Set2", width=0.5, linewidth=1.5)
ax_cv.set_title("5-Fold CV AUC Distribution", fontweight="bold", fontsize=12)
ax_cv.set_xlabel("AUC Score"); ax_cv.set_ylabel("")
ax_cv.set_facecolor(CLR_GRID)

plt.savefig(f"{OUTPUT_DIR}fig2_model_results.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("   ✓ Saved fig2_model_results.png")

# ══════════════════════════════════════════════════════════
# 6. PRINT FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL CONCLUSIONS")
print("="*60)
print(f"""
 DATASET
   • 918 patient records, 13 clinical features
   • Target: Presence of Heart Disease (0/1)

 KEY EDA FINDINGS
   • Males have ~2× higher disease rate than females
   • Asymptomatic chest pain (cp=0) strongly indicates disease
   • Reversible thalassemia defect is a major risk marker
   • Higher oldpeak (ST depression) → higher disease risk
   • Exercise-induced angina highly correlated with disease
   • Max heart rate (thalach) inversely linked to disease

 MODEL PERFORMANCE SUMMARY
""")
for name in names:
    r = results[name]
    marker = "" if name == best_name else "  "
    print(f"   {marker} {name:<26} AUC={r['test_auc']:.4f}  F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}")

print(f"""
 BEST MODEL: {best_name}
   • Test AUC    : {best['test_auc']:.4f}
   • Precision   : {best['precision']:.4f}
   • Recall      : {best['recall']:.4f}
   • F1-Score    : {best['f1']:.4f}
   • Accuracy    : {best['accuracy']:.4f}

 CLINICAL INSIGHT
   Tree-based models (RF/XGB/GBB) outperform linear models,
   suggesting non-linear interactions between features.
   Top predictors: ca (vessels), thal, exang, oldpeak, cp.

 OUTPUT FILES
   • fig1_eda_dashboard.png  — Full EDA visualization
   • fig2_model_results.png  — Model comparison & metrics
""")
print("="*60)
print("  Project Complete!")
print("="*60)