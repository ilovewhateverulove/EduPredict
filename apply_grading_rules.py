"""
EduPredict — apply_grading_rules.py
════════════════════════════════════════════════════════════════════
Re-grades data/Student_Performance_Enhanced.xlsx with the app's current
subject rule (subject_outcome in app.py):

  Pass  = carry ≥ 30/60 AND final ≥ 20/40 AND not barred (> 5 absences)
  Grade = from the total /100, capped at D when carry or final is below 50%

Updated columns: every Sem*_<Subject>_Grade / _GradePoint / _PassFail,
Sem*_SubjectsFailed, Sem*_CGPA, TotalSubjectsFailed and AvgCGPA.
Marks, totals and attendance are not changed.

Usage:
  python apply_grading_rules.py --dry-run     # show what would change
  python apply_grading_rules.py               # write the file

Afterwards retrain so the models and predictions.csv follow the new rule:
  python train_model.py
  python train_model_sem3.py
════════════════════════════════════════════════════════════════════
"""

import argparse
import os

os.environ["AUTO_INIT_DB"] = "false"     # only the grading helpers are needed, not the database

import pandas as pd
from app import subject_outcome, CGPA_FLOOR, MAX_ABSENCES_ALLOWED, SUBJECTS

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = os.path.join(BASE_DIR, "data", "Student_Performance_Enhanced.xlsx")
SEMS      = ("Sem1", "Sem2", "Sem3")


def regrade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for sem in SEMS:
        points, failed = [], 0
        for subject in SUBJECTS:
            p = f"{sem}_{subject}_"
            barred = df[p + "Absences"] > MAX_ABSENCES_ALLOWED
            final_exam = df[p + "FinalExam"].where(~barred, 0)
            results = [subject_outcome(c, f, t, b) for c, f, t, b in
                       zip(df[p + "CarryTotal"], final_exam, df[p + "Total"], barred)]
            df[p + "Grade"] = [g for g, _, _ in results]
            df[p + "GradePoint"] = [pt for _, pt, _ in results]
            df[p + "PassFail"] = ["Pass" if ok else "Fail" for _, _, ok in results]
            points.append(df[p + "GradePoint"])
            failed = failed + (df[p + "PassFail"] == "Fail").astype(int)
        df[f"{sem}_SubjectsFailed"] = failed
        df[f"{sem}_CGPA"] = (sum(points) / len(points)).clip(CGPA_FLOOR, 4.0).round(2)
    df["TotalSubjectsFailed"] = sum(df[f"{sem}_SubjectsFailed"] for sem in SEMS)
    df["AvgCGPA"] = (sum(df[f"{sem}_CGPA"] for sem in SEMS) / len(SEMS)).round(2)
    return df


def main():
    parser = argparse.ArgumentParser(description="Re-grade the dataset with the current pass/fail rule.")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = parser.parse_args()

    sheets = pd.read_excel(DATA_XLSX, sheet_name=None)
    first = next(iter(sheets))
    old = sheets[first]
    new = regrade(old)

    pf_cols = [f"{s}_{j}_PassFail" for s in SEMS for j in SUBJECTS]
    gp_cols = [f"{s}_{j}_GradePoint" for s in SEMS for j in SUBJECTS]
    print(f"  Subjects failed : {int((old[pf_cols] == 'Fail').sum().sum()):>5} → {int((new[pf_cols] == 'Fail').sum().sum())}")
    print(f"  Grades lowered  : {int((new[gp_cols] < old[gp_cols]).sum().sum()):>5}")
    print(f"  AvgCGPA changed : {int(((new['AvgCGPA'] - old['AvgCGPA']).abs() > 0.005).sum()):>5} students "
          f"(mean {old['AvgCGPA'].mean():.2f} → {new['AvgCGPA'].mean():.2f})")

    if args.dry_run:
        print("\n  Dry run: nothing written.")
        return
    sheets[first] = new
    with pd.ExcelWriter(DATA_XLSX) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    print(f"\n  [OK] {os.path.relpath(DATA_XLSX, BASE_DIR)} updated. Now run train_model.py and train_model_sem3.py.")


if __name__ == "__main__":
    main()
