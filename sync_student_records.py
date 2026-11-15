"""
EduPredict — sync_student_records.py
════════════════════════════════════════════════════════════════════
Refreshes students already in the database after the dataset is
re-graded or the models are retrained:

  - students in outputs/predictions.csv get its grades, pass/fail,
    CGPA, failed counts and new risk prediction
  - students added by hand (not in the CSV) are re-graded from their
    own stored marks with the current rule; their risk is left as is

Sections, names, emails, interventions and lecturers are not touched,
and no risk-change emails are sent (this is a bulk re-grade, not a
real change in a student's situation).

Usage:
  python sync_student_records.py --dry-run
  python sync_student_records.py
════════════════════════════════════════════════════════════════════
"""

import argparse
import os

import pandas as pd

from app      import app, derive_academic_summary, SUBJECTS
from database import db, Student, Prediction, Assessment, AttendanceRecord

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_CSV = os.path.join(BASE_DIR, "outputs", "predictions.csv")


def main():
    parser = argparse.ArgumentParser(description="Refresh student records from predictions.csv.")
    parser.add_argument("--dry-run", action="store_true", help="report changes without saving")
    args = parser.parse_args()

    df = pd.read_csv(PRED_CSV)
    rows = {str(r["StudentID"]).strip(): r for _, r in df.iterrows()}

    with app.app_context():
        from_csv = regraded = risk_changed = 0
        for student in Student.query.all():
            pred = Prediction.query.filter_by(student_id=student.student_id).first()
            assessments = {(a.semester, a.subject): a
                           for a in Assessment.query.filter_by(student_id=student.student_id)}
            row = rows.get(student.student_id)

            if row is not None:
                for sem in (1, 2, 3):
                    for subject in SUBJECTS:
                        a = assessments.get((sem, subject))
                        if a:
                            prefix = f"Sem{sem}_{subject}_"
                            a.grade = row[prefix + "Grade"]
                            a.grade_point = float(row[prefix + "GradePoint"])
                            a.pass_fail = row[prefix + "PassFail"]
                if pred:
                    for sem in (1, 2, 3):
                        setattr(pred, f"sem{sem}_cgpa", float(row[f"Sem{sem}_CGPA"]))
                    pred.avg_cgpa = float(row["AvgCGPA"])
                    pred.total_failed = int(row["TotalSubjectsFailed"])
                    risk_changed += pred.risk_level != row["risk_level"]
                    pred.risk_probability = float(row["risk_probability"])
                    pred.prediction_label = row["prediction_label"]
                    pred.risk_level = row["risk_level"]
                from_csv += 1
                continue

            # Not in the dataset: re-grade from the student's own marks
            data = {}
            for (sem, subject), a in assessments.items():
                att = AttendanceRecord.query.filter_by(student_id=student.student_id,
                                                       semester=sem, subject=subject).first()
                prefix = f"Sem{sem}_{subject}_"
                data.update({prefix + "Quiz": a.quiz, prefix + "Assignment": a.assignment,
                             prefix + "FinalExam": a.final_exam,
                             prefix + "Absences": att.absences if att else 0})
            derived = derive_academic_summary(data)
            for key, result in derived["subjects"].items():
                a = assessments[key]
                a.grade, a.grade_point = result["grade"], result["grade_point"]
                a.pass_fail = "Pass" if result["passed"] else "Fail"
            summary = derived["summary"]
            if pred and summary:
                for sem in derived["semesters"]:
                    setattr(pred, f"sem{sem}_cgpa", summary[f"Sem{sem}_CGPA"])
                pred.avg_cgpa = summary["AvgCGPA"]
                pred.total_failed = summary["TotalFailedSubjects"]
            regraded += 1

        print(f"  {from_csv} students refreshed from predictions.csv ({risk_changed} changed risk level)")
        print(f"  {regraded} students not in the dataset re-graded from their own marks")
        if args.dry_run:
            db.session.rollback()
            print("\n  Dry run: nothing saved.")
            return
        db.session.commit()
        print("\n  [OK] Database updated.")


if __name__ == "__main__":
    main()
