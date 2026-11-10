"""
EduPredict — train_model.py
═══════════════════════════════════════════════════════════════════════════
IIUM Assessment & Attendance Rules:
  • 14 lectures per subject per semester
  • Max 5 absences per subject — exceed this = BARRED from final exam
  • Barred students automatically fail that subject (Final Exam = 0)
  • Max 15 total absences per semester (3 subjects × 5)
  • Barring threshold: attendance < 64.3% per subject (9/14 classes)

  Subject pass conditions:
    • Carry Total (Quiz+Assignment) ≥ 24/60 (≥ 40% of carry marks)
    • Final Exam ≥ 16/40            (≥ 40% of exam marks)
    • Subject Total ≥ 50/100

  Grade & GPA Point table:
    80–100 → A   4.00    75–79 → A-  3.67    70–74 → B+  3.33
    65–69  → B   3.00    60–64 → B-  2.67    55–59 → C+  2.33
    50–54  → C   2.00    45–49 → D   1.67    40–44 → D-  1.33
    35–39  → E   1.00     0–34 → F   0.00

  CGPA formula:
    Sem CGPA = (Math_point + Science_point + English_point) / 3
    Avg CGPA = (Sem1_CGPA + Sem2_CGPA + Sem3_CGPA) / 3
    Range    = 1.50 – 4.00 (clipped)
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

SUBJECTS            = ['Math', 'Science', 'English']
SEMS                = ['Sem1', 'Sem2', 'Sem3']
CLASSES_PER_SUBJECT = 14
MAX_ABS_PER_SUBJECT = 5
TOTAL_CLASSES_SEM   = 42   # 3 × 14

print('=' * 64)
print('  EduPredict — Semester 4 At-Risk Prediction (IIUM)')
print('=' * 64)


#  STEP 1 — LOAD DATASET
print('\n[1/6] Loading dataset...')
df = pd.read_excel(DATA_PATH)
print(f'      {df.shape[0]:,} students × {df.shape[1]} columns loaded')
print(f'      CGPA range    : {df["AvgCGPA"].min():.2f} – {df["AvgCGPA"].max():.2f}')
print(f'      Attendance    : {df["OverallAttendancePct"].min():.1f}% – {df["OverallAttendancePct"].max():.1f}%')
barred_any = (df['TotalBarredCount'] > 0).sum()
print(f'      Ever barred   : {barred_any:,} students ({barred_any/len(df)*100:.1f}%)')


#  STEP 2 — FEATURE ENGINEERING
print('\n[2/6] Engineering features...')

# CGPA features 
df['CGPATrend']     = (df['Sem3_CGPA'] - df['Sem1_CGPA']).round(4)
df['CGPADrop']      = (df[['Sem1_CGPA','Sem2_CGPA']].max(axis=1) - df['Sem3_CGPA']).round(4)
df['LowCGPA']       = (df['AvgCGPA'] < 2.5).astype(int)
df['DecliningCGPA'] = (df['CGPATrend'] < -0.3).astype(int)

# Attendance & barring features 
df['LowAttSem3']    = (df['Sem3_AttendancePct'] < 64.3).astype(int)  # IIUM threshold
df['LowAttOverall'] = (df['OverallAttendancePct'] < 64.3).astype(int)
df['AttDropping']   = (df['AttendanceTrend'] < -5).astype(int)

# Absence rate per sem (0–1, max 15 absences / 42 classes)
for sem in SEMS:
    df[f'AbsRate_{sem}'] = (df[f'{sem}_TotalAbsences'] / TOTAL_CLASSES_SEM).clip(0, 1).round(4)

# Per-subject barring risk — strongest signal for Sem 4
for sem in SEMS:
    for subj in SUBJECTS:
        df[f'{sem}_{subj}_AtRisk'] = (df[f'{sem}_{subj}_Absences'] >= MAX_ABS_PER_SUBJECT).astype(int)
    # Count subjects at/near barring limit this semester
    df[f'{sem}_NearBarCount'] = sum(
        df[f'{sem}_{subj}_AtRisk'] for subj in SUBJECTS
    )

# Was student barred in Sem 3? (most recent — strongest predictor for Sem 4)
df['BarredSem3']     = (df['Sem3_BarredCount'] > 0).astype(int)
df['BarredMultiple'] = (df['TotalBarredCount'] >= 2).astype(int)

# Grade-point based features 
for sem in SEMS:
    gp_cols = [f'{sem}_{subj}_GradePoint' for subj in SUBJECTS]
    df[f'{sem}_MinGradePoint']   = df[gp_cols].min(axis=1).round(2)
    df[f'{sem}_LowSubjectCount'] = sum(
        (df[f'{sem}_{subj}_GradePoint'] < 2.00).astype(int) for subj in SUBJECTS
    )

df['CarryTrend']     = (df['Sem3_TotalCarry'] - df['Sem1_TotalCarry']).round(1)
df['FinalTrend']     = (df['Sem3_TotalFinal'] - df['Sem1_TotalFinal']).round(1)
df['LowCarrySem3']   = (df['Sem3_TotalCarry'] < 72).astype(int)
df['LowFinalSem3']   = (df['Sem3_TotalFinal'] < 48).astype(int)
df['MultiFailSem3']  = (df['Sem3_SubjectsFailed'] >= 2).astype(int)
df['ConsistentFail'] = (df['TotalSubjectsFailed'] >= 3).astype(int)

# Encode categoricals 
CAT_COLS = ['Gender','Ethnicity','ParentalEducation','StudyMethod',
            'Tutoring','Extracurricular','Sports','Music','Volunteering']
le = LabelEncoder()
for col in CAT_COLS:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

print(f'      Feature engineering complete')


#  STEP 3 — DEFINE SEM 4 AT-RISK LABEL
print('\n[3/6] Defining Semester 4 at-risk label...')

# At-Risk for Sem 4 if ANY:
#  Rule 1 — AvgCGPA < 2.50 (weak academic standing)
#  Rule 2 — Sem3 attendance < 64.3% (at/near IIUM barring threshold)
#  Rule 3 — Attendance declining AND overall below 75%
#  Rule 4 — Barred from at least 1 subject in Sem 3 (most recent)
#  Rule 5 — Failed 2+ subjects in Sem 3
#  Rule 6 — Low carry marks in Sem 3 (coursework weakness)
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
rules = [
    ('Low AvgCGPA (<2.50)',               df['AvgCGPA'] < 2.5),
    ('Low Sem3 Attendance (<64.3%)',       df['Sem3_AttendancePct'] < 64.3),
    ('Attendance declining',               (df['AttendanceTrend']<-10)&(df['OverallAttendancePct']<75)),
    ('Barred in Sem3',                     df['Sem3_BarredCount'] > 0),
    ('Failed 2+ subjects in Sem3',         df['Sem3_SubjectsFailed'] >= 2),
    ('Low carry marks Sem3',               df['LowCarrySem3'] == 1),
]
for i, (label, cond) in enumerate(rules, 1):
    print(f'        Rule {i} — {label}: {cond.sum():,}')


#  STEP 4 — FEATURE LIST & SPLIT
print('\n[4/6] Preparing features and splitting data...')

FEATURES = [
    # Attendance per semester
    'Sem1_AttendancePct','Sem2_AttendancePct','Sem3_AttendancePct',
    'OverallAttendancePct','AttendanceTrend',
    'LowAttSem3','LowAttOverall','AttDropping',
    'AbsRate_Sem1','AbsRate_Sem2','AbsRate_Sem3',

    # Absences and barring (IIUM-specific)
    'Sem1_TotalAbsences','Sem2_TotalAbsences','Sem3_TotalAbsences',
    'Sem1_BarredCount','Sem2_BarredCount','Sem3_BarredCount',
    'TotalBarredCount','BarredSem3','BarredMultiple',
    'Sem1_NearBarCount','Sem2_NearBarCount','Sem3_NearBarCount',

    # CGPA
    'Sem1_CGPA','Sem2_CGPA','Sem3_CGPA',
    'AvgCGPA','CGPATrend','CGPADrop','LowCGPA','DecliningCGPA',

    # Per-subject GPA points (Sem3 most predictive)
    'Sem3_Math_GradePoint','Sem3_Science_GradePoint','Sem3_English_GradePoint',
    'Sem2_Math_GradePoint','Sem2_Science_GradePoint','Sem2_English_GradePoint',
    'Sem1_Math_GradePoint','Sem1_Science_GradePoint','Sem1_English_GradePoint',
    'Sem1_MinGradePoint','Sem2_MinGradePoint','Sem3_MinGradePoint',
    'Sem1_LowSubjectCount','Sem2_LowSubjectCount','Sem3_LowSubjectCount',

    # Carry and final totals
    'Sem1_TotalCarry','Sem1_TotalFinal','Sem1_TotalMarks',
    'Sem2_TotalCarry','Sem2_TotalFinal','Sem2_TotalMarks',
    'Sem3_TotalCarry','Sem3_TotalFinal','Sem3_TotalMarks',
    'CarryTrend','FinalTrend','LowCarrySem3','LowFinalSem3',

    # Subject failures
    'Sem1_SubjectsFailed','Sem2_SubjectsFailed','Sem3_SubjectsFailed',
    'TotalSubjectsFailed','MultiFailSem3','ConsistentFail',

    # Study behaviour
    'StudyTimeWeekly',

    # Encoded categoricals
    'Gender_enc','Ethnicity_enc','ParentalEducation_enc',
    'StudyMethod_enc','Tutoring_enc',
    'Extracurricular_enc','Sports_enc','Music_enc','Volunteering_enc',
]

X = df[FEATURES]
y = df['at_risk']
# 70% train / 15% validation / 15% test — test is held back until final evaluation
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
)
print(f'      Total features : {len(FEATURES)}')
print(f'      Training rows  : {len(X_train):,}  (70%)')
print(f'      Validation rows: {len(X_val):,}  (15%)')
print(f'      Test rows      : {len(X_test):,}  (15%)')


#  STEP 5 — TRAIN MODEL
print('\n[5/6] Training XGBoost model...')
model = XGBClassifier(
    n_estimators=1000, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
    early_stopping_rounds=50, random_state=42, n_jobs=-1
)
# Stops adding trees once validation logloss has not improved for 50 rounds
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
best_trees = model.best_iteration + 1
print(f'      Early stopping chose {best_trees} trees')

val_pred  = model.predict(X_val)
val_acc   = accuracy_score(y_val, val_pred)
val_f1    = f1_score(y_val, val_pred)
val_auc   = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

# Cross-validation on train + validation only, so test rows never influence it
cv_model = XGBClassifier(**{**model.get_params(),
                            'n_estimators': best_trees, 'early_stopping_rounds': None})
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(cv_model, X_temp, y_temp, cv=cv, scoring='accuracy', n_jobs=-1)
cv_f1  = cross_val_score(cv_model, X_temp, y_temp, cv=cv, scoring='f1',       n_jobs=-1)
cv_auc = cross_val_score(cv_model, X_temp, y_temp, cv=cv, scoring='roc_auc',  n_jobs=-1)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc     = accuracy_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)

print('\n' + '=' * 64)
print('  MODEL EVALUATION')
print('=' * 64)
print(f'\n  Validation set (15%, used for early stopping):')
print(f'    Accuracy  : {val_acc:.4f}  ({val_acc*100:.2f}%)')
print(f'    F1 Score  : {val_f1:.4f}')
print(f'    ROC-AUC   : {val_auc:.4f}')
print(f'\n  Test set (holdout 15%, final result):')
print(f'    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)')
print(f'    F1 Score  : {f1:.4f}')
print(f'    ROC-AUC   : {auc:.4f}')
print(f'\n  5-Fold Cross-Validation (train + validation, test excluded):')
print(f'    Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}')
print(f'    F1 Score  : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}')
print(f'    ROC-AUC   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}')
print(f'\n  Confusion matrix:')
print(f'    True Negative  : {cm[0,0]}')
print(f'    False Positive : {cm[0,1]}')
print(f'    False Negative : {cm[1,0]}')
print(f'    True Positive  : {cm[1,1]}')
print()
print(classification_report(y_test, y_pred, target_names=['Safe','At-Risk']))

fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print('\n  Top 15 feature importances:')
for i, (feat, imp) in enumerate(fi.head(15).items(), 1):
    bar = '█' * int(imp * 100)
    print(f'    {i:2}. {feat:<38} {bar} {imp:.4f}')


#  STEP 6 — SAVE EVERYTHING
print('\n[6/6] Saving outputs...')
joblib.dump(model,    os.path.join(MODEL_DIR, 'model.pkl'))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, 'features.pkl'))
print('      ✓ model.pkl + features.pkl → model/')

# Feature importance chart
top_fi  = fi.head(15).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
colors  = ['#DC2626' if v >= fi.quantile(0.85) else
           '#D97706' if v >= fi.quantile(0.5)  else '#1565C0' for v in top_fi.values]
bars    = ax.barh(top_fi.index, top_fi.values, color=colors, edgecolor='none', height=0.65)
ax.set_xlabel('Feature Importance', fontsize=11, color='#64748B')
ax.set_title('Top 15 Feature Importances — Sem 4 At-Risk Prediction (IIUM)',
             fontsize=12, fontweight='bold', color='#0B1E3D', pad=14)
ax.tick_params(axis='both', labelsize=9, colors='#64748B')
ax.spines[['top','right','bottom']].set_visible(False)
ax.set_facecolor('white'); fig.patch.set_facecolor('white')
ax.xaxis.grid(True, color='#F0F3F9', linewidth=1); ax.set_axisbelow(True)
for bar, val in zip(bars, top_fi.values):
    ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, color='#64748B')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print('      ✓ feature_importance.png → outputs/')

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm,
    display_labels=['Safe','At-Risk']).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Sem 4 Prediction (IIUM)',
             fontsize=12, fontweight='bold', color='#0B1E3D', pad=12)
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print('      ✓ confusion_matrix.png → outputs/')

# predictions.csv
def risk_level(prob):
    if prob >= 0.70: return 'High Risk'
    elif prob >= 0.40: return 'Medium Risk'
    return 'Low Risk'

df['risk_probability'] = model.predict_proba(X)[:, 1]
df['prediction_label'] = ['At-Risk' if p==1 else 'Safe' for p in model.predict(X)]
df['risk_level']       = df['risk_probability'].apply(risk_level)

output_cols = ['StudentID','Age','Gender','Ethnicity','ParentalEducation',
               'StudyTimeWeekly','StudyMethod','Tutoring']
for sem in SEMS:
    for subj in SUBJECTS:
        output_cols += [
            f'{sem}_{subj}_Absences', f'{sem}_{subj}_Attendance',
            f'{sem}_{subj}_Barred',
            f'{sem}_{subj}_Quiz', f'{sem}_{subj}_Assignment',
            f'{sem}_{subj}_CarryTotal', f'{sem}_{subj}_FinalExam',
            f'{sem}_{subj}_Total', f'{sem}_{subj}_Grade',
            f'{sem}_{subj}_GradePoint', f'{sem}_{subj}_PassFail',
        ]
    output_cols += [f'{sem}_TotalAbsences', f'{sem}_AttendancePct',
                    f'{sem}_TotalCarry', f'{sem}_TotalFinal',
                    f'{sem}_TotalMarks', f'{sem}_SubjectsFailed',
                    f'{sem}_BarredCount', f'{sem}_CGPA']
output_cols += [
    'TotalAbsences','OverallAttendancePct','AttendanceTrend',
    'TotalSubjectsFailed','TotalBarredCount',
    'Extracurricular','Sports','Music','Volunteering',
    'AvgCGPA','at_risk','prediction_label','risk_probability','risk_level',
]
df[output_cols].to_csv(os.path.join(OUT_DIR, 'predictions.csv'), index=False)
print('      ✓ predictions.csv → outputs/')

rl = df['risk_level'].value_counts()
print(f'\n  Predicted Sem 4 risk levels:')
print(f'    High Risk   : {rl.get("High Risk",   0):,}')
print(f'    Medium Risk : {rl.get("Medium Risk", 0):,}')
print(f'    Low Risk    : {rl.get("Low Risk",    0):,}')
print('\n' + '=' * 64)
print('  Done. Run python app.py to start EduPredict.')
print('=' * 64 + '\n')
