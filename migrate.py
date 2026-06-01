"""
EduPredict — migrate.py
════════════════════════════════════════════════════════════════════
Run this ONCE after setting up your database to import
predictions.csv into the database tables.

Usage:
  python migrate.py

Works with both SQLite and MySQL — whichever DATABASE_URL
is set in your .env file.
════════════════════════════════════════════════════════════════════
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app      import app, init_db
from database import db, Student, Prediction, Assessment, AttendanceRecord

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PRED_CSV     = os.path.join(BASE_DIR, "outputs", "predictions.csv")

SUBJECTS = ["Math", "Science", "English"]
SEMS     = [1, 2, 3]
SEM_NAMES = {1: "Sem1", 2: "Sem2", 3: "Sem3"}


def get_val(row, col, default=0):
    """Safely get a value from a CSV row, returning default if missing/NaN."""
    if col not in row or pd.isna(row[col]):
        return default
    return row[col]


def migrate():
    print("=" * 56)
    print("  EduPredict — Database Migration")
    print("=" * 56)

    if not os.path.isfile(PRED_CSV):
        print(f"\n  ERROR: {PRED_CSV} not found.")
        print("  Run train_model.py first to generate predictions.csv")
        return

    with app.app_context():
        # Create tables if they don't exist
        init_db()

        df = pd.read_csv(PRED_CSV)
        df["StudentID"] = df["StudentID"].astype(str).str.strip()

        total     = len(df)
        imported  = 0
        skipped   = 0

        print(f"\n  Importing {total:,} students from predictions.csv ...\n")

        for idx, row in df.iterrows():
            sid = str(row["StudentID"])

            # ── Skip if student already exists ─────────────────────────────
            if Student.query.filter_by(student_id=sid).first():
                skipped += 1
                continue

            # ── Student profile ────────────────────────────────────────────
            student = Student(
                student_id         = sid,
                age                = int(get_val(row, "Age", 20)),
                gender             = get_val(row, "Gender", "Male"),
                ethnicity          = get_val(row, "Ethnicity", "Malay"),
                parental_education = get_val(row, "ParentalEducation", "Secondary School (SPM)"),
                study_hours        = float(get_val(row, "StudyTimeWeekly", 10)),
                study_method       = get_val(row, "StudyMethod", "Mixed"),
                tutoring           = get_val(row, "Tutoring", "No"),
                extracurricular    = get_val(row, "Extracurricular", "No"),
                sports             = get_val(row, "Sports", "No"),
                music              = get_val(row, "Music", "No"),
                volunteering       = get_val(row, "Volunteering", "No"),
            )
            db.session.add(student)

            # ── Prediction (ML output) ─────────────────────────────────────
            pred = Prediction(
                student_id       = sid,
                sem1_cgpa        = float(get_val(row, "Sem1_CGPA", 0)),
                sem2_cgpa        = float(get_val(row, "Sem2_CGPA", 0)),
                sem3_cgpa        = float(get_val(row, "Sem3_CGPA", 0)),
                avg_cgpa         = float(get_val(row, "AvgCGPA", 0)),
                sem1_att         = float(get_val(row, "Sem1_AttendancePct", 0)),
                sem2_att         = float(get_val(row, "Sem2_AttendancePct", 0)),
                sem3_att         = float(get_val(row, "Sem3_AttendancePct", 0)),
                overall_att      = float(get_val(row, "OverallAttendancePct", 0)),
                att_trend        = float(get_val(row, "AttendanceTrend", 0)),
                sem1_barred      = int(get_val(row, "Sem1_BarredCount", 0)),
                sem2_barred      = int(get_val(row, "Sem2_BarredCount", 0)),
                sem3_barred      = int(get_val(row, "Sem3_BarredCount", 0)),
                total_barred     = int(get_val(row, "TotalBarredCount", 0)),
                total_failed     = int(get_val(row, "TotalSubjectsFailed", 0)),
                risk_probability = float(get_val(row, "risk_probability", 0)),
                prediction_label = get_val(row, "prediction_label", "Safe"),
                risk_level       = get_val(row, "risk_level", "Low Risk"),
            )
            db.session.add(pred)

            # ── Assessment + Attendance (per subject per semester) ──────────
            for sem_n in SEMS:
                sem_name = SEM_NAMES[sem_n]
                for subj in SUBJECTS:
                    # Assessment
                    a = Assessment(
                        student_id  = sid,
                        semester    = sem_n,
                        subject     = subj,
                        quiz        = float(get_val(row, f"{sem_name}_{subj}_Quiz", 0)),
                        assignment  = float(get_val(row, f"{sem_name}_{subj}_Assignment", 0)),
                        carry_total = float(get_val(row, f"{sem_name}_{subj}_CarryTotal", 0)),
                        final_exam  = float(get_val(row, f"{sem_name}_{subj}_FinalExam", 0)),
                        total       = float(get_val(row, f"{sem_name}_{subj}_Total", 0)),
                        grade       = get_val(row, f"{sem_name}_{subj}_Grade", "—"),
                        grade_point = float(get_val(row, f"{sem_name}_{subj}_GradePoint", 0)),
                        pass_fail   = get_val(row, f"{sem_name}_{subj}_PassFail", "—"),
                    )
                    db.session.add(a)

                    # Attendance
                    barred_val = get_val(row, f"{sem_name}_{subj}_Barred", "No")
                    att = AttendanceRecord(
                        student_id = sid,
                        semester   = sem_n,
                        subject    = subj,
                        absences   = int(get_val(row, f"{sem_name}_{subj}_Absences", 0)),
                        attendance = float(get_val(row, f"{sem_name}_{subj}_Attendance", 0)),
                        barred     = (barred_val == "Yes"),
                    )
                    db.session.add(att)

            imported += 1

            # Commit every 100 rows to avoid memory overload
            if imported % 100 == 0:
                db.session.commit()
                print(f"  ... {imported:,} / {total:,} imported")

        # Final commit
        db.session.commit()

        print(f"\n  ✓ Migration complete!")
        print(f"    Imported : {imported:,} students")
        print(f"    Skipped  : {skipped:,}  (already existed)")
        print(f"    Database : {os.environ.get('DATABASE_URL', 'sqlite:///edupredict.db')}")
        print()


if __name__ == "__main__":
    migrate()