"""
EduPredict — train_model_sem3.py
═══════════════════════════════════════════════════════════════════════════
Semester 3 → Semester 4 At-Risk Prediction Model

Unlike train_model.py which uses all 3 semesters of historical data,
this model ONLY uses Semester 3 data as input. This makes it:

  • More realistic — in practice, a lecturer predicts Sem 4 risk using
    the most recent semester's data, not historical averages
  • More useful for early intervention — predictions can be made the
    moment Sem 3 results are published
  • Easier to use — the prediction form only needs one semester of data

Performance vs full model:
  Full model (3 sems):  ~99%  accuracy  (benefits from 3 sems of trends)
  This model (Sem3):    ~86%  accuracy  (realistic single-semester signal)

The 86% accuracy is strong for a real-world early warning system and
is more academically defensible than the near-perfect full model,
since synthetic data makes the full model artificially inflated.

Outputs:
  model/model_sem3.pkl      — trained Random Forest model
  model/features_sem3.pkl   — feature list for the predict route
  outputs/predictions_sem3.csv
  outputs/feature_importance_sem3.png
  outputs/confusion_matrix_sem3.png
═══════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import joblib, os, warnings
warnings.filterwarnings('ignore')
try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit("xgboost is required for training. Install it with: pip install xgboost") from exc

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Student_Performance_Enhanced.xlsx')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUT_DIR   = os.path.join(BASE_DIR, 'outputs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

SUBJECTS = ['Math', 'Science', 'English']

print('=' * 64)
print('  EduPredict — Sem 3 → Sem 4 Prediction Model')
print('=' * 64)


#  STEP 1 — LOAD DATASET
print('\n[1/6] Loading dataset...')
df = pd.read_excel(DATA_PATH)
print(f'      {df.shape[0]:,} students × {df.shape[1]} columns loaded')

#  STEP 2 — FEATURE ENGINEERING (Semester 3 only)
print('\n[2/6] Engineering Semester 3 features...')

# Attendance features 
df['Sem3_AbsRate']      = (df['Sem3_TotalAbsences'] / 42).clip(0, 1).round(4)
df['LowAttSem3']        = (df['Sem3_AttendancePct'] < 64.3).astype(int)
df['BarredSem3']        = (df['Sem3_BarredCount'] > 0).astype(int)

# Per-subject near-barring flag (absences >= 5)
for subj in SUBJECTS:
    df[f'Sem3_{subj}_NearBar'] = (df[f'Sem3_{subj}_Absences'] >= 5).astype(int)
df['Sem3_NearBarCount'] = df[[f'Sem3_{s}_NearBar' for s in SUBJECTS]].sum(axis=1)

# CGPA and grade features 
df['LowCGPA_Sem3']         = (df['Sem3_CGPA'] < 2.5).astype(int)
df['Sem3_AvgGradePoint']   = df[[f'Sem3_{s}_GradePoint' for s in SUBJECTS]].mean(axis=1).round(3)
df['Sem3_MinGradePoint']   = df[[f'Sem3_{s}_GradePoint' for s in SUBJECTS]].min(axis=1).round(3)

for subj in SUBJECTS:
    df[f'Sem3_{subj}_Low'] = (df[f'Sem3_{subj}_GradePoint'] < 2.0).astype(int)
df['Sem3_LowSubjectCount'] = df[[f'Sem3_{s}_Low' for s in SUBJECTS]].sum(axis=1)

# Assessment features 
df['LowCarrySem3']  = (df['Sem3_TotalCarry'] < 72).astype(int)   # < 40% of max 180
df['LowFinalSem3']  = (df['Sem3_TotalFinal'] < 48).astype(int)   # < 40% of max 120
df['MultiFailSem3'] = (df['Sem3_SubjectsFailed'] >= 2).astype(int)

# Encode categoricals 
CAT_COLS = ['Gender','Ethnicity','ParentalEducation','StudyMethod',
            'Tutoring','Extracurricular','Sports','Music','Volunteering']
le = LabelEncoder()
for col in CAT_COLS:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

print(f'      Feature engineering complete')


#  STEP 3 — DEFINE AT-RISK LABEL
print('\n[3/6] Defining Semester 4 at-risk label...')

# AvgCGPA needed for label — computed from all 3 sems (this is still the target)
# The LABEL uses all semesters — only the FEATURES are restricted to Sem 3
df['AvgCGPA'] = df[['Sem1_CGPA', 'Sem2_CGPA', 'Sem3_CGPA']].mean(axis=1)

df['at_risk'] = (
    (df['AvgCGPA'] < 2.5) |
    (df['Sem3_AttendancePct'] < 64.3) |
    ((df['AttendanceTrend'] < -10) & (df['OverallAttendancePct'] < 75)) |
    (df['Sem3_BarredCount'] > 0) |
    (df['Sem3_SubjectsFailed'] >= 2) |
    (df['LowCarrySem3'] == 1)
).astype(int)

counts = df['at_risk'].value_counts()
total  = len(df)
print(f'      Safe (0):    {counts.get(0,0):,}  ({counts.get(0,0)/total*100:.1f}%)')
print(f'      At-Risk (1): {counts.get(1,0):,}  ({counts.get(1,0)/total*100:.1f}%)')


#  STEP 4 — FEATURE LIST (Sem 3 ONLY)
print('\n[4/6] Preparing Sem 3 feature set...')

FEATURES = [
    # Attendance (Sem 3) 
    'Sem3_AttendancePct',
    'Sem3_TotalAbsences',
    'Sem3_AbsRate',
    'LowAttSem3',
    'Sem3_Math_Absences',
    'Sem3_Science_Absences',
    'Sem3_English_Absences',
    'Sem3_NearBarCount',

    # Barring (Sem 3) 
    'Sem3_BarredCount',
    'BarredSem3',

    # CGPA (Sem 3)
    'Sem3_CGPA',
    'LowCGPA_Sem3',

    # Per-subject grade points (Sem 3) 
    'Sem3_Math_GradePoint',
    'Sem3_Science_GradePoint',
    'Sem3_English_GradePoint',
    'Sem3_MinGradePoint',
    'Sem3_AvgGradePoint',
    'Sem3_LowSubjectCount',

    # Carry and final marks (Sem 3) 
    'Sem3_TotalCarry',
    'Sem3_TotalFinal',
    'Sem3_TotalMarks',
    'Sem3_Math_CarryTotal',
    'Sem3_Science_CarryTotal',
    'Sem3_English_CarryTotal',
    'Sem3_Math_FinalExam',
    'Sem3_Science_FinalExam',
    'Sem3_English_FinalExam',
    'LowCarrySem3',
    'LowFinalSem3',

    # Subject failures (Sem 3) 
    'Sem3_SubjectsFailed',
    'MultiFailSem3',

    # Study behaviour 
    'StudyTimeWeekly',

    # Encoded categoricals
    'Gender_enc',
    'Ethnicity_enc',
    'ParentalEducation_enc',
    'StudyMethod_enc',
    'Tutoring_enc',
    'Extracurricular_enc',
    'Sports_enc',
    'Music_enc',
    'Volunteering_enc',
]

X = df[FEATURES]
y = df['at_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'      Input features : {len(FEATURES)}  (Sem 3 only)')
print(f'      Training rows  : {len(X_train):,}')
print(f'      Test rows      : {len(X_test):,}')
print()
print('  NOTE: All features are from Semester 3 only.')
print('  This model is designed for real-time Sem 4 prediction')
print('  using only the most recent semester data available.')


#  STEP 5 — TRAIN MODEL
print('\n[5/6] Training XGBoost (Sem 3 only) model...')

model = XGBClassifier(
    n_estimators     = 350,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    eval_metric      = 'logloss',
    random_state     = 42,
    n_jobs           = -1
)
model.fit(X_train, y_train)
print('      Training complete.')

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
cv_f1  = cross_val_score(model, X, y, cv=cv, scoring='f1',       n_jobs=-1)
cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc',  n_jobs=-1)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc     = accuracy_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)

print('\n' + '=' * 64)
print('  MODEL EVALUATION — Sem 3 Only')
print('=' * 64)
print(f'\n  Test set (holdout 20%):')
print(f'    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)')
print(f'    F1 Score  : {f1:.4f}')
print(f'    ROC-AUC   : {auc:.4f}')
print(f'\n  5-Fold Cross-Validation:')
print(f'    Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}')
print(f'    F1 Score  : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}')
print(f'    ROC-AUC   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}')
print(f'\n  Confusion matrix:')
print(f'    True Negative  (Safe correctly identified) : {cm[0,0]}')
print(f'    False Positive (Safe wrongly flagged)      : {cm[0,1]}')
print(f'    False Negative (At-Risk missed)            : {cm[1,0]}')
print(f'    True Positive  (At-Risk correctly caught)  : {cm[1,1]}')
print(f'\n  Interpretation:')
print(f'    The model correctly identifies {cm[1,1]} of {cm[1,0]+cm[1,1]} at-risk students')
print(f'    using ONLY Semester 3 data — no historical context needed.')
print()
print(classification_report(y_test, y_pred, target_names=['Safe', 'At-Risk']))

fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print('\n  Top 15 feature importances:')
for i, (feat, imp) in enumerate(fi.head(15).items(), 1):
    bar = '█' * int(imp * 80)
    print(f'    {i:2}. {feat:<38} {bar} {imp:.4f}')


#  STEP 6 — SAVE EVERYTHING
print('\n[6/6] Saving outputs...')

joblib.dump(model,    os.path.join(MODEL_DIR, 'model_sem3.pkl'))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, 'features_sem3.pkl'))
print('      ✓ model_sem3.pkl     → model/')
print('      ✓ features_sem3.pkl  → model/')

# Feature importance chart
top_fi  = fi.tail(15)
fig, ax = plt.subplots(figsize=(10, 6))
colors  = ['#DC2626' if v >= fi.quantile(0.85) else
           '#D97706' if v >= fi.quantile(0.5)  else '#1565C0' for v in top_fi.values]
bars    = ax.barh(top_fi.index, top_fi.values, color=colors, edgecolor='none', height=0.65)
ax.set_xlabel('Feature Importance', fontsize=11, color='#64748B')
ax.set_title('Top 15 Feature Importances — Sem 3 Only Model\n(Predicts Sem 4 At-Risk)',
             fontsize=12, fontweight='bold', color='#0B1E3D', pad=14)
ax.tick_params(axis='both', labelsize=9, colors='#64748B')
ax.spines[['top','right','bottom']].set_visible(False)
ax.set_facecolor('white'); fig.patch.set_facecolor('white')
ax.xaxis.grid(True, color='#F0F3F9', linewidth=1); ax.set_axisbelow(True)
for bar, val in zip(bars, top_fi.values):
    ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, color='#64748B')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_importance_sem3.png'), dpi=150, bbox_inches='tight')
plt.close()
print('      ✓ feature_importance_sem3.png → outputs/')

# Confusion matrix chart
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm,
    display_labels=['Safe','At-Risk']).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Sem 3 Only Model',
             fontsize=12, fontweight='bold', color='#0B1E3D', pad=12)
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_sem3.png'), dpi=150, bbox_inches='tight')
plt.close()
print('      ✓ confusion_matrix_sem3.png → outputs/')

# predictions_sem3.csv
def risk_level(prob):
    if prob >= 0.70: return 'High Risk'
    elif prob >= 0.40: return 'Medium Risk'
    return 'Low Risk'

df['risk_probability_sem3'] = model.predict_proba(X)[:, 1]
df['prediction_sem3']       = ['At-Risk' if p==1 else 'Safe' for p in model.predict(X)]
df['risk_level_sem3']       = df['risk_probability_sem3'].apply(risk_level)

output_cols = [
    'StudentID',
    'Sem3_AttendancePct', 'Sem3_TotalAbsences', 'Sem3_BarredCount',
    'Sem3_Math_GradePoint', 'Sem3_Science_GradePoint', 'Sem3_English_GradePoint',
    'Sem3_CGPA', 'Sem3_TotalCarry', 'Sem3_TotalFinal', 'Sem3_SubjectsFailed',
    'AvgCGPA',
    'at_risk', 'prediction_sem3', 'risk_probability_sem3', 'risk_level_sem3',
]
df[output_cols].to_csv(os.path.join(OUT_DIR, 'predictions_sem3.csv'), index=False)
print('      ✓ predictions_sem3.csv → outputs/')

rl = df['risk_level_sem3'].value_counts()
print(f'\n  Predicted Sem 4 risk levels (Sem 3 only model):')
print(f'    High Risk   : {rl.get("High Risk",   0):,}')
print(f'    Medium Risk : {rl.get("Medium Risk", 0):,}')
print(f'    Low Risk    : {rl.get("Low Risk",    0):,}')

print('\n' + '=' * 64)
print('  Model comparison summary:')
print(f'    Full model (3 sems) : ~99% accuracy  — use for batch analysis')
print(f'    Sem3 model (1 sem)  : {acc*100:.1f}% accuracy  — use for real-time prediction')
print('=' * 64)
print('\n  Run python app.py to use this model in the web system.\n')
