"""
EduPredict — app.py
════════════════════════════════════════════════════════════════════
Supports two database backends via DATABASE_URL in .env:
  SQLite  (local/FYP)   → sqlite:///edupredict.db
  MySQL   (production)  → mysql+pymysql://user:pass@host/edupredict

Run locally:  python app.py
Deploy:       set .env variables on your hosting panel
════════════════════════════════════════════════════════════════════
"""

from flask     import Flask, render_template, request, redirect, url_for, jsonify, session, Response, abort
from functools import wraps
from dotenv    import load_dotenv
from database  import (db, bcrypt, Student, Prediction, LivePrediction,
                       AdminUser, Assessment, AttendanceRecord,
                       Lecturer, ClassSection, RiskNotification)
import joblib
import pandas  as pd
import os
import secrets
from datetime  import datetime
from werkzeug.utils import secure_filename
from sqlalchemy import inspect, true, false

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

# Gemini AI — configure once at startup 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
        print(f"  [OK] Gemini AI configured (model: {GEMINI_MODEL}, key: ...{GEMINI_API_KEY[-6:]})")
    except Exception as e:
        genai = None
        GEMINI_ENABLED = False
        print(f"  [WARN] Gemini import failed: {e}")
else:
    genai = None
    GEMINI_ENABLED = False
    print("  [WARN] GEMINI_API_KEY not found in .env - AI chat disabled.")

app = Flask(__name__)

#  Config
app.secret_key = os.environ.get("SECRET_KEY", "edupredict-2025-s3cr3t-k3y-do-not-share")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///edupredict.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 280,
    "pool_pre_ping": True,
}

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "edupredict123")
# Shared by all students for the FYP demo. Student login stays off if this is unset.
DEMO_STUDENT_PASSWORD = os.environ.get("DEMO_STUDENT_PASSWORD", "").strip()

db.init_app(app)
bcrypt.init_app(app)

# ML models 
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "sections")
SECTION_MODEL_DIR = os.path.join(BASE_DIR, "model", "sections")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SECTION_MODEL_DIR, exist_ok=True)

# Full model (all 3 semesters) — used by student dashboard batch predictions
model        = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
FEATURES     = joblib.load(os.path.join(BASE_DIR, "model", "features.pkl"))

# Sem3-only model — used by manual prediction form (/predict)
model_sem3   = joblib.load(os.path.join(BASE_DIR, "model", "model_sem3.pkl"))
FEATURES_SEM3= joblib.load(os.path.join(BASE_DIR, "model", "features_sem3.pkl"))

# Encoding maps 
GENDER_MAP    = {"Male": 1, "Female": 0}
ETHNICITY_MAP = {"Bumiputera Sabah/Sarawak": 0, "Chinese": 1, "Indian": 2, "Malay": 3}
EDU_MAP = {
    "Bachelor's Degree": 0, "Diploma": 1, "Master's Degree": 2,
    "No Formal Education": 3, "PhD": 4,
    "Primary School (UPSR)": 5, "Secondary School (SPM)": 6
}
METHOD_MAP = {"Group Study": 0, "Mixed": 1, "Notes": 2, "Online Videos": 3, "Textbook": 4}
YESNO_MAP  = {"No": 0, "Yes": 1}

# Grade table (matches train_model.py) 
GRADE_TABLE = [
    (80,100,4.00),(75,79,3.67),(70,74,3.33),(65,69,3.00),(60,64,2.67),
    (55,59,2.33),(50,54,2.00),(45,49,1.67),(40,44,1.33),(35,39,1.00),(0,34,0.00),
]

def get_grade_point(total: float) -> float:
    t = round(float(total))
    for lo, hi, point in GRADE_TABLE:
        if lo <= t <= hi:
            return point
    return 0.00

TOTAL_CLASSES_SEM = 42   # 3 subjects × 14 classes


def build_predict_features(raw: dict) -> pd.DataFrame:
    """
    Build the Sem3-only feature set from raw form inputs.
    Each subject (Math, Science, English) has its OWN carry marks —
    Quiz + Assignment are collected separately per subject, not shared.
    """
    att3  = raw["att3"]
    abs3  = raw["abs3"]
    cgpa3 = raw["sem3"]

    # Per-subject carry marks (Quiz + Assignment each subject)
    math_quiz  = raw["math_quiz"];  math_assn  = raw["math_assn"]
    sci_quiz   = raw["sci_quiz"];   sci_assn   = raw["sci_assn"]
    eng_quiz   = raw["eng_quiz"];   eng_assn   = raw["eng_assn"]

    math_carry = round(math_quiz + math_assn, 1)   # /60
    sci_carry  = round(sci_quiz  + sci_assn,  1)   # /60
    eng_carry  = round(eng_quiz  + eng_assn,  1)   # /60

    # Per-subject final exam marks
    math_final = raw["math_final"]   # /40
    sci_final  = raw["sci_final"]    # /40
    eng_final  = raw["eng_final"]    # /40

    # Per-subject totals
    math_total = round(math_carry + math_final, 1)   # /100
    sci_total  = round(sci_carry  + sci_final,  1)   # /100
    eng_total  = round(eng_carry  + eng_final,  1)   # /100

    # Per-subject absences (default: distribute total evenly if not provided)
    math_abs = raw.get("math_abs3", abs3 / 3)
    sci_abs  = raw.get("sci_abs3",  abs3 / 3)
    eng_abs  = raw.get("eng_abs3",  abs3 / 3)

    # Aggregates
    total_carry = round(math_carry + sci_carry + eng_carry, 1)   # /180
    total_final = round(math_final + sci_final + eng_final, 1)   # /120
    total_marks = round(math_total + sci_total + eng_total, 1)   # /300

    # Grade points from total marks per subject
    def gp(total):
        t = round(float(total))
        for lo, hi, point in GRADE_TABLE:
            if lo <= t <= hi: return point
        return 0.0

    math_gp = gp(math_total)
    sci_gp  = gp(sci_total)
    eng_gp  = gp(eng_total)

    # Barring: >5 absences per subject = barred from final
    barred_cnt = sum(1 for a in [math_abs, sci_abs, eng_abs] if a > 5)
    near_bar   = sum(1 for a in [math_abs, sci_abs, eng_abs] if a >= 5)

    # Subject failures: fail if carry<24 OR final<16 OR total<50
    def is_fail(carry, final, total):
        return carry < 24 or final < 16 or total < 50
    failed = sum(1 for c, f, t in [
        (math_carry, math_final, math_total),
        (sci_carry,  sci_final,  sci_total),
        (eng_carry,  eng_final,  eng_total),
    ] if is_fail(c, f, t))

    d = {
        "Sem3_AttendancePct":       att3,
        "Sem3_TotalAbsences":       int(abs3),
        "Sem3_AbsRate":             round(min(abs3 / TOTAL_CLASSES_SEM, 1), 4),
        "LowAttSem3":               int(att3 < 64.3),
        "Sem3_Math_Absences":       round(math_abs),
        "Sem3_Science_Absences":    round(sci_abs),
        "Sem3_English_Absences":    round(eng_abs),
        "Sem3_NearBarCount":        near_bar,
        "Sem3_BarredCount":         barred_cnt,
        "BarredSem3":               int(barred_cnt > 0),
        "Sem3_CGPA":                cgpa3,
        "LowCGPA_Sem3":             int(cgpa3 < 2.5),
        "Sem3_Math_GradePoint":     math_gp,
        "Sem3_Science_GradePoint":  sci_gp,
        "Sem3_English_GradePoint":  eng_gp,
        "Sem3_MinGradePoint":       min(math_gp, sci_gp, eng_gp),
        "Sem3_AvgGradePoint":       round((math_gp + sci_gp + eng_gp) / 3, 3),
        "Sem3_LowSubjectCount":     sum(1 for p in [math_gp, sci_gp, eng_gp] if p < 2.0),
        "Sem3_TotalCarry":          total_carry,
        "Sem3_TotalFinal":          total_final,
        "Sem3_TotalMarks":          total_marks,
        "Sem3_Math_CarryTotal":     math_carry,
        "Sem3_Science_CarryTotal":  sci_carry,
        "Sem3_English_CarryTotal":  eng_carry,
        "Sem3_Math_FinalExam":      math_final,
        "Sem3_Science_FinalExam":   sci_final,
        "Sem3_English_FinalExam":   eng_final,
        "LowCarrySem3":             int(total_carry < 72),
        "LowFinalSem3":             int(total_final < 48),
        "Sem3_SubjectsFailed":      failed,
        "MultiFailSem3":            int(failed >= 2),
        "StudyTimeWeekly":          raw["study_hours"],
        "Gender_enc":               GENDER_MAP.get(raw["gender"], 0),
        "Ethnicity_enc":            ETHNICITY_MAP.get(raw["ethnicity"], 0),
        "ParentalEducation_enc":    EDU_MAP.get(raw["education"], 0),
        "StudyMethod_enc":          METHOD_MAP.get(raw["method"], 0),
        "Tutoring_enc":             YESNO_MAP.get(raw["tutoring"], 0),
        "Extracurricular_enc":      YESNO_MAP.get(raw["extracurricular"], 0),
        "Sports_enc":               YESNO_MAP.get(raw["sports"], 0),
        "Music_enc":                YESNO_MAP.get(raw["music"], 0),
        "Volunteering_enc":         YESNO_MAP.get(raw["volunteering"], 0),
    }
    return pd.DataFrame([d])[FEATURES_SEM3]


def manual_prediction_factors(raw: dict, risk_prob: float) -> list[dict]:
    subjects = [
        ("Math", raw["math_quiz"] + raw["math_assn"], raw["math_final"],
         raw["math_quiz"] + raw["math_assn"] + raw["math_final"], raw.get("math_abs3", 0)),
        ("Science", raw["sci_quiz"] + raw["sci_assn"], raw["sci_final"],
         raw["sci_quiz"] + raw["sci_assn"] + raw["sci_final"], raw.get("sci_abs3", 0)),
        ("English", raw["eng_quiz"] + raw["eng_assn"], raw["eng_final"],
         raw["eng_quiz"] + raw["eng_assn"] + raw["eng_final"], raw.get("eng_abs3", 0)),
    ]
    factors = []
    if risk_prob >= 0.70:
        factors.append({"level": "High", "text": f"Risk probability is {risk_prob * 100:.1f}%, above the high-risk threshold."})
    elif risk_prob >= 0.40:
        factors.append({"level": "Medium", "text": f"Risk probability is {risk_prob * 100:.1f}%, above the medium-risk threshold."})
    if raw["att3"] < 64.3:
        factors.append({"level": "High", "text": f"Semester 3 attendance is {raw['att3']:.1f}%, below the 64.3% minimum."})
    elif raw["att3"] < 75:
        factors.append({"level": "Medium", "text": f"Semester 3 attendance is only {raw['att3']:.1f}%, which reduces confidence in passing."})
    if raw["abs3"] >= 12:
        factors.append({"level": "High", "text": f"Total absences are high at {raw['abs3']}/42 classes."})
    elif raw["abs3"] >= 8:
        factors.append({"level": "Medium", "text": f"Total absences are elevated at {raw['abs3']}/42 classes."})
    if raw["sem3"] < 2.5:
        factors.append({"level": "High", "text": f"Semester 3 CGPA is {raw['sem3']:.2f}, below the 2.50 risk threshold."})
    elif raw["sem3"] < 3.0:
        factors.append({"level": "Medium", "text": f"Semester 3 CGPA is {raw['sem3']:.2f}, close to the lower performance band."})
    failed = []
    barred = []
    weak_carry = []
    weak_final = []
    for name, carry, final_exam, total, absences in subjects:
        if absences > 5:
            barred.append(f"{name} ({absences:.0f} absences)")
        if carry < 24:
            weak_carry.append(f"{name} ({carry:.1f}/60)")
        if final_exam < 16:
            weak_final.append(f"{name} ({final_exam:.1f}/40)")
        if carry < 24 or final_exam < 16 or total < 50:
            failed.append(f"{name} ({total:.1f}/100)")
    if barred:
        factors.append({"level": "High", "text": "Barring risk detected in " + ", ".join(barred) + "."})
    if failed:
        factors.append({"level": "High", "text": "Predicted weak/failing subject totals: " + ", ".join(failed) + "."})
    if weak_carry:
        factors.append({"level": "Medium", "text": "Low carry marks in " + ", ".join(weak_carry) + "."})
    if weak_final:
        factors.append({"level": "Medium", "text": "Low final exam marks in " + ", ".join(weak_final) + "."})
    if not factors:
        factors.append({"level": "Low", "text": "No major academic or attendance risk factor was detected from the submitted values."})
    return factors[:6]



#  DB INIT
def init_db():
    with app.app_context():
        db.create_all()
        ensure_sqlite_columns()
        if not AdminUser.query.filter_by(username=ADMIN_USERNAME).first():
            admin = AdminUser(username=ADMIN_USERNAME)
            admin.set_password(ADMIN_PASSWORD)
            db.session.add(admin)
            print(f"  Admin '{ADMIN_USERNAME}' created.")
        ensure_default_section()
        db.session.flush()
        seed_demo_students_from_predictions()
        db.session.commit()



#  HELPERS
def get_risk_level(prob: float) -> str:
    if prob >= 0.70:   return "High Risk"
    elif prob >= 0.40: return "Medium Risk"
    return "Low Risk"


RISK_ORDER = {"Low Risk": 1, "Medium Risk": 2, "High Risk": 3}


def current_lecturer_id():
    return session.get("lecturer_id") if session.get("lecturer_logged_in") else None


def lecturer_sections(include_archived=False):
    lecturer_id = current_lecturer_id()
    if not lecturer_id:
        return []
    q = ClassSection.query.filter_by(lecturer_id=lecturer_id)
    if not include_archived:
        q = q.filter_by(is_archived=False)
    return q.order_by(ClassSection.section_name.asc()).all()


def get_active_section():
    sections = lecturer_sections()
    if not sections:
        return None
    active_id = session.get("active_section_id")
    section = next((s for s in sections if s.id == active_id), None)
    if not section:
        section = sections[0]
        session["active_section_id"] = section.id
    return section


def scoped_student_query():
    q = Student.query
    section_ids = [s.id for s in lecturer_sections()]
    if current_lecturer_id():
        q = q.filter(Student.section_id.in_(section_ids or [None]))
    return q


def scoped_prediction_query():
    q = Prediction.query
    section_ids = [s.id for s in lecturer_sections()]
    if current_lecturer_id():
        q = (q.join(Student, Student.student_id == Prediction.student_id)
             .filter(Student.section_id.in_(section_ids or [None])))
    return q


def staff_can_access_student(student: Student) -> bool:
    if session.get("admin_logged_in"):
        return True
    if not student or not current_lecturer_id():
        return False
    return any(section.id == student.section_id for section in lecturer_sections())


def lecturer_section_filter():
    return Student.section_id.in_([s.id for s in lecturer_sections()] or [None]) if current_lecturer_id() else true()


def detect_semester_depth(columns) -> int:
    names = [str(c) for c in columns]
    if any("Sem3" in c for c in names):
        return 3
    if any("Sem2" in c for c in names):
        return 2
    return 1


def model_confidence(sem_depth: int) -> dict:
    return {
        3: {"label": "High", "pct": 90, "based_on": "Semester 1, 2, and 3 data"},
        2: {"label": "Medium", "pct": 72, "based_on": "Semester 1 and 2 data"},
        1: {"label": "Low", "pct": 48, "based_on": "Semester 1 data only"},
    }.get(sem_depth, {"label": "Unavailable", "pct": 0, "based_on": "No semester history"})


def expected_template_columns(sem_depth=3):
    cols = [
        "StudentID", "Age", "Gender", "Ethnicity", "ParentalEducation",
        "StudyTimeWeekly", "StudyMethod", "Tutoring", "Extracurricular",
        "Sports", "Music", "Volunteering",
    ]
    for sem in range(1, int(sem_depth) + 1):
        cols += [f"Sem{sem}_CGPA", f"Sem{sem}_AttendancePct"]
        for subject in SUBJECTS:
            cols += [
                f"Sem{sem}_{subject}_Quiz",
                f"Sem{sem}_{subject}_Assignment",
                f"Sem{sem}_{subject}_FinalExam",
                f"Sem{sem}_{subject}_Absences",
            ]
    cols += ["RiskProbability", "PredictionLabel", "RiskLevel"]
    return cols


def ensure_default_section():
    lecturer = Lecturer.query.filter_by(username=ADMIN_USERNAME).first()
    if not lecturer:
        lecturer = Lecturer(username=ADMIN_USERNAME, full_name="Default Lecturer")
        lecturer.set_password(ADMIN_PASSWORD)
        db.session.add(lecturer)
        db.session.flush()
    section = ClassSection.query.filter_by(
        lecturer_id=lecturer.id, section_name="Default Section").first()
    if not section:
        section = ClassSection(
            lecturer_id=lecturer.id,
            section_name="Default Section",
            current_semester=3,
            sem_depth=3,
            model_path=os.path.join(BASE_DIR, "model", "model_sem3.pkl"),
            features_path=os.path.join(BASE_DIR, "model", "features_sem3.pkl"),
        )
        db.session.add(section)
        db.session.flush()
    Student.query.filter(Student.section_id.is_(None)).update(
        {Student.section_id: section.id, Student.registration_status: "preloaded"})


def _seed_val(row, col, default=0):
    if col not in row or pd.isna(row[col]):
        return default
    return row[col]


def seed_demo_students_from_predictions():
    if Student.query.count() > 0:
        return
    csv_path = os.path.join(BASE_DIR, "outputs", "predictions.csv")
    if not os.path.isfile(csv_path):
        print(f"  Demo seed skipped: {csv_path} not found.")
        return

    default_section = ClassSection.query.filter_by(section_name="Default Section").first()
    df = pd.read_csv(csv_path)
    df["StudentID"] = df["StudentID"].astype(str).str.strip()
    imported = 0

    for _, row in df.iterrows():
        sid = str(row["StudentID"]).strip()
        if not sid or sid.lower() == "nan":
            continue

        student = Student(
            student_id=sid,
            section_id=default_section.id if default_section else None,
            registration_status="preloaded",
            age=int(_seed_val(row, "Age", 20)),
            gender=_seed_val(row, "Gender", "Male"),
            ethnicity=_seed_val(row, "Ethnicity", "Malay"),
            parental_education=_seed_val(row, "ParentalEducation", "Secondary School (SPM)"),
            study_hours=float(_seed_val(row, "StudyTimeWeekly", 10)),
            study_method=_seed_val(row, "StudyMethod", "Mixed"),
            tutoring=_seed_val(row, "Tutoring", "No"),
            extracurricular=_seed_val(row, "Extracurricular", "No"),
            sports=_seed_val(row, "Sports", "No"),
            music=_seed_val(row, "Music", "No"),
            volunteering=_seed_val(row, "Volunteering", "No"),
        )
        db.session.add(student)

        db.session.add(Prediction(
            student_id=sid,
            sem1_cgpa=float(_seed_val(row, "Sem1_CGPA", 0)),
            sem2_cgpa=float(_seed_val(row, "Sem2_CGPA", 0)),
            sem3_cgpa=float(_seed_val(row, "Sem3_CGPA", 0)),
            avg_cgpa=float(_seed_val(row, "AvgCGPA", 0)),
            sem1_att=float(_seed_val(row, "Sem1_AttendancePct", 0)),
            sem2_att=float(_seed_val(row, "Sem2_AttendancePct", 0)),
            sem3_att=float(_seed_val(row, "Sem3_AttendancePct", 0)),
            overall_att=float(_seed_val(row, "OverallAttendancePct", 0)),
            att_trend=float(_seed_val(row, "AttendanceTrend", 0)),
            sem1_barred=int(_seed_val(row, "Sem1_BarredCount", 0)),
            sem2_barred=int(_seed_val(row, "Sem2_BarredCount", 0)),
            sem3_barred=int(_seed_val(row, "Sem3_BarredCount", 0)),
            total_barred=int(_seed_val(row, "TotalBarredCount", 0)),
            total_failed=int(_seed_val(row, "TotalSubjectsFailed", 0)),
            risk_probability=float(_seed_val(row, "risk_probability", 0)),
            prediction_label=_seed_val(row, "prediction_label", "Safe"),
            risk_level=_seed_val(row, "risk_level", "Low Risk"),
        ))

        for sem_n, sem_name in ((1, "Sem1"), (2, "Sem2"), (3, "Sem3")):
            for subj in SUBJECTS:
                db.session.add(Assessment(
                    student_id=sid,
                    semester=sem_n,
                    subject=subj,
                    quiz=float(_seed_val(row, f"{sem_name}_{subj}_Quiz", 0)),
                    assignment=float(_seed_val(row, f"{sem_name}_{subj}_Assignment", 0)),
                    carry_total=float(_seed_val(row, f"{sem_name}_{subj}_CarryTotal", 0)),
                    final_exam=float(_seed_val(row, f"{sem_name}_{subj}_FinalExam", 0)),
                    total=float(_seed_val(row, f"{sem_name}_{subj}_Total", 0)),
                    grade=_seed_val(row, f"{sem_name}_{subj}_Grade", "-"),
                    grade_point=float(_seed_val(row, f"{sem_name}_{subj}_GradePoint", 0)),
                    pass_fail=_seed_val(row, f"{sem_name}_{subj}_PassFail", "-"),
                ))
                barred_val = _seed_val(row, f"{sem_name}_{subj}_Barred", "No")
                db.session.add(AttendanceRecord(
                    student_id=sid,
                    semester=sem_n,
                    subject=subj,
                    absences=int(_seed_val(row, f"{sem_name}_{subj}_Absences", 0)),
                    attendance=float(_seed_val(row, f"{sem_name}_{subj}_Attendance", 0)),
                    barred=(barred_val == "Yes"),
                ))

        imported += 1
        if imported % 100 == 0:
            db.session.commit()

    db.session.commit()
    print(f"  Demo seed imported {imported} students from outputs/predictions.csv.")


def ensure_sqlite_columns():
    if not app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite"):
        return
    inspector = inspect(db.engine)
    existing = {c["name"] for c in inspector.get_columns("students")}
    additions = {
        "section_id": "INTEGER",
        "email": "VARCHAR(120)",
        "password_hash": "VARCHAR(255)",
        "registration_status": "VARCHAR(20) DEFAULT 'pending'",
    }
    with db.engine.begin() as conn:
        for name, ddl in additions.items():
            if name not in existing:
                conn.exec_driver_sql(f"ALTER TABLE students ADD COLUMN {name} {ddl}")


def make_section_classifier():
    if XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ), "XGBoost"
    return GradientBoostingClassifier(random_state=42), "GradientBoosting fallback"


def prepare_training_frame(df: pd.DataFrame, sem_depth: int):
    df = df.copy()
    target_col = None
    for candidate in ("at_risk", "AtRisk", "PredictionLabel", "Prediction"):
        if candidate in df.columns:
            target_col = candidate
            break
    if not target_col:
        raise ValueError("CSV must include PredictionLabel/Prediction or at_risk for training.")
    y_raw = df[target_col]
    y_num = pd.to_numeric(y_raw, errors="coerce")
    if y_num.notna().all():
        y = y_num.fillna(0).astype(int)
    else:
        y = y_raw.astype(str).str.lower().isin(["at-risk", "atrisk", "1", "true", "yes"]).astype(int)
    blocked = {target_col, "RiskLevel", "RiskProbability", "risk_probability", "student_id", "StudentID", "Student_ID", "id"}
    feature_df = df.drop(columns=[c for c in blocked if c in df.columns], errors="ignore")
    feature_df = feature_df[[c for c in feature_df.columns if f"Sem{sem_depth + 1}" not in str(c)]]
    feature_df = pd.get_dummies(feature_df, dummy_na=True)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    if feature_df.empty:
        raise ValueError("No usable feature columns found after CSV validation.")
    return feature_df, y


def train_model_for_section(csv_path: str, section: ClassSection):
    df = pd.read_csv(csv_path)
    sem_depth = detect_semester_depth(df.columns)
    X, y = prepare_training_frame(df, sem_depth)
    if y.nunique() < 2:
        raise ValueError("Training data needs both Safe and At-Risk examples.")
    clf, model_name = make_section_classifier()
    clf.fit(X, y)
    safe_section = secure_filename(section.section_name) or f"section_{section.id}"
    stem = f"model_section_{section.id}_{safe_section}_sem{sem_depth}"
    model_path = os.path.join(SECTION_MODEL_DIR, f"{stem}.pkl")
    features_path = os.path.join(SECTION_MODEL_DIR, f"{stem}_features.pkl")
    joblib.dump(clf, model_path)
    joblib.dump(list(X.columns), features_path)
    section.sem_depth = sem_depth
    section.model_path = model_path
    section.features_path = features_path
    section.updated_at = datetime.utcnow()
    return model_name


def maybe_notify_risk_change(student: Student, pred: Prediction, new_risk_level: str):
    old = pred.risk_level if pred else None
    if old == new_risk_level or RISK_ORDER.get(new_risk_level, 0) <= RISK_ORDER.get(old, 0):
        return
    msg = (
        f"Your EduPredict risk level changed from {old or 'Unknown'} to {new_risk_level}. "
        "Please review your dashboard and contact your lecturer if you need support."
    )
    db.session.add(RiskNotification(
        student_id=student.student_id,
        section_id=student.section_id,
        old_risk_level=old,
        new_risk_level=new_risk_level,
        recipient=student.email,
        message=msg,
        sent=bool(student.email),
    ))


SUBJECTS = ["Math", "Science", "English"]


def build_dashboard_stats(student_q, pred_q):
    total   = student_q.count()
    at_risk = pred_q.filter(Prediction.prediction_label == "At-Risk").count()
    safe    = pred_q.filter(Prediction.prediction_label == "Safe").count()
    high    = pred_q.filter(Prediction.risk_level == "High Risk").count()
    medium  = pred_q.filter(Prediction.risk_level == "Medium Risk").count()
    low     = pred_q.filter(Prediction.risk_level == "Low Risk").count()
    avg_c   = pred_q.with_entities(db.func.avg(Prediction.avg_cgpa)).scalar()    or 0
    avg_a   = pred_q.with_entities(db.func.avg(Prediction.overall_att)).scalar() or 0
    numeric_ids = []
    for s in student_q.with_entities(Student.student_id).all():
        sid = str(s.student_id)
        if sid.isdigit():
            numeric_ids.append(int(sid))
    return {
        "total":          total,
        "at_risk":        at_risk,
        "safe":           safe,
        "at_risk_pct":    round(at_risk / total * 100, 1) if total else 0,
        "safe_pct":       round(safe    / total * 100, 1) if total else 0,
        "avg_cgpa":       round(float(avg_c), 2),
        "avg_attendance": round(float(avg_a), 1),
        "high_risk":      high,
        "medium_risk":    medium,
        "low_risk":       low,
        "current_student_id": max(numeric_ids) if numeric_ids else 1001,
    }


def build_student_dict(student: Student, pred: Prediction) -> dict:
    """Build context dict for dashboard.html from ORM objects.
    NOTE: local dict is named sem_data (not assessment) to avoid
    shadowing the Assessment model class imported above.
    """
    SUBJECTS = ["Math", "Science", "English"]
    SEMS     = ["Sem1", "Sem2", "Sem3"]

    sem_data = {}
    for sem in SEMS:
        sem_n    = int(sem[-1])
        cgpa_val = getattr(pred, f"sem{sem_n}_cgpa", 0) or 0
        sem_data[sem] = {
            "cgpa":           round(float(cgpa_val), 2),
            "subjects":       [],
            "subjects_failed": 0,
            "barred_count":   0,
            "total_absences": 0,
            "attendance_pct": round(float(getattr(pred, f"sem{sem_n}_att", 0) or 0), 1),
            "total_carry":    0.0,
            "total_final":    0.0,
            "total_marks":    0.0,
        }
        for subj in SUBJECTS:
            a   = Assessment.query.filter_by(
                      student_id=student.student_id,
                      semester=sem_n, subject=subj).first()
            att = AttendanceRecord.query.filter_by(
                      student_id=student.student_id,
                      semester=sem_n, subject=subj).first()

            subj_data = {
                "name":       subj,
                "absences":   att.absences if att else 0,
                "attendance": round(float(att.attendance or 0), 1) if att else 0,
                "barred":     "Yes" if (att and att.barred) else "No",
                "quiz":       round(float(a.quiz        or 0), 1)  if a else 0,
                "assignment": round(float(a.assignment  or 0), 1)  if a else 0,
                "carry":      round(float(a.carry_total or 0), 1)  if a else 0,
                "final":      round(float(a.final_exam  or 0), 1)  if a else 0,
                "total":      round(float(a.total       or 0), 1)  if a else 0,
                "grade":      a.grade                              if a else "—",
                "point":      round(float(a.grade_point or 0), 2)  if a else 0,
                "status":     a.pass_fail                          if a else "—",
            }
            sem_data[sem]["subjects"].append(subj_data)

            if a   and a.pass_fail == "Fail": sem_data[sem]["subjects_failed"] += 1
            if att and att.barred:            sem_data[sem]["barred_count"]    += 1
            if att:                           sem_data[sem]["total_absences"]  += att.absences or 0
            if a:
                sem_data[sem]["total_carry"] += a.carry_total or 0
                sem_data[sem]["total_final"] += a.final_exam  or 0
                sem_data[sem]["total_marks"] += a.total       or 0
        sem_data[sem]["total_carry"] = round(float(sem_data[sem]["total_carry"]), 1)
        sem_data[sem]["total_final"] = round(float(sem_data[sem]["total_final"]), 1)
        sem_data[sem]["total_marks"] = round(float(sem_data[sem]["total_marks"]), 1)

    return {
        "id":              student.student_id,
        "age":             student.age or 0,
        "gender":          student.gender            or "—",
        "ethnicity":       student.ethnicity          or "—",
        "education":       student.parental_education or "—",
        "sem1_attendance": round(float(pred.sem1_att   or 0), 1),
        "sem2_attendance": round(float(pred.sem2_att   or 0), 1),
        "sem3_attendance": round(float(pred.sem3_att   or 0), 1),
        "overall_att":     round(float(pred.overall_att or 0), 1),
        "att_trend":       round(float(pred.att_trend  or 0), 1),
        "sem1_absences":   0, "sem2_absences": 0, "sem3_absences": 0,
        "total_barred":    pred.total_barred or 0,
        "study_hours":     round(float(student.study_hours or 0), 1),
        "study_method":    student.study_method    or "—",
        "tutoring":        student.tutoring         or "—",
        "extracurricular": student.extracurricular  or "—",
        "sports":          student.sports           or "—",
        "music":           student.music            or "—",
        "volunteering":    student.volunteering      or "—",
        "sem1_cgpa":       round(float(pred.sem1_cgpa or 0), 2),
        "sem2_cgpa":       round(float(pred.sem2_cgpa or 0), 2),
        "sem3_cgpa":       round(float(pred.sem3_cgpa or 0), 2),
        "avg_cgpa":        round(float(pred.avg_cgpa  or 0), 2),
        "total_failed":    pred.total_failed or 0,
        "prediction":      pred.prediction_label,
        "risk_prob":       round(float(pred.risk_probability) * 100, 1),
        "risk_level":      pred.risk_level,
        "assessment":      sem_data,
        "cgpa_trend": [
            {"sem": "Sem 1", "cgpa": round(float(pred.sem1_cgpa or 0), 2)},
            {"sem": "Sem 2", "cgpa": round(float(pred.sem2_cgpa or 0), 2)},
            {"sem": "Sem 3", "cgpa": round(float(pred.sem3_cgpa or 0), 2)},
        ],
    }



#  AUTH DECORATOR
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Rule 1: any active student session = go home, no admin access ever
        if session.get("student_id"):
            return redirect(url_for("home"))
        if not session.get("admin_logged_in") and not session.get("lecturer_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("student_id"):
            return redirect(url_for("home"))
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


def lecturer_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("student_id"):
            return redirect(url_for("home"))
        if not session.get("lecturer_logged_in"):
            return redirect(url_for("lecturer_login"))
        return f(*args, **kwargs)
    return decorated



#  AUTH ROUTES
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    # Students cannot access admin login at all
    if session.get("student_id"):
        return redirect(url_for("home"))
    # Always show the login form — never auto-redirect based on session
    # Clear any stale admin session so re-authentication is always required
    if request.method == "GET":
        session.pop("admin_logged_in", None)
        session.pop("admin_user", None)
        session.pop("lecturer_logged_in", None)
        session.pop("lecturer_id", None)
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        admin    = AdminUser.query.filter_by(username=username).first()
        if admin and admin.check_password(password):
            session.clear()                        # wipe everything first
            session["admin_logged_in"] = True
            session["admin_user"]      = username
            return redirect(url_for("admin"))
        error = "Invalid username or password. Please try again."
    return render_template("admin_login.html", error=error, login_role="admin")


@app.route("/lecturer/login", methods=["GET", "POST"])
def lecturer_login():
    if session.get("student_id"):
        return redirect(url_for("home"))
    if request.method == "GET":
        session.pop("admin_logged_in", None)
        session.pop("admin_user", None)
        session.pop("lecturer_logged_in", None)
        session.pop("lecturer_id", None)
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        lecturer = Lecturer.query.filter_by(username=username).first()
        if lecturer and lecturer.check_password(password):
            session.clear()
            session["lecturer_logged_in"] = True
            session["lecturer_id"] = lecturer.id
            session["lecturer_user"] = username
            return redirect(url_for("lecturer_dashboard"))
        error = "Invalid lecturer username or password. Please try again."
    return render_template("admin_login.html", error=error, login_role="lecturer")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.route("/lecturer/logout")
def lecturer_logout():
    session.clear()
    return redirect(url_for("lecturer_login"))


@app.route("/student/logout")
def student_logout():
    session.pop("student_id", None)
    return redirect(url_for("home"))



#  STUDENT ROUTES
@app.route("/", methods=["GET", "POST"])
def home():
    session.pop("student_id", None)
    session.pop("admin_logged_in", None)   # prevent admin session persisting to home
    session.pop("admin_user", None)
    session.pop("lecturer_id", None)
    session.pop("lecturer_logged_in", None)
    session.pop("lecturer_user", None)
    error = None
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        password = request.form.get("password", "")
        if not DEMO_STUDENT_PASSWORD:
            error = "Student login is not configured. Please contact the administrator."
        elif not student_id or not password:
            error = "Please enter your Student ID and password."
        else:
            student = Student.query.filter_by(student_id=student_id).first()
            password_ok = secrets.compare_digest(password.encode(), DEMO_STUDENT_PASSWORD.encode())
            if not student or not password_ok:
                error = "Incorrect Student ID or password. Please check and try again."
            else:
                session["student_id"] = student_id
                return redirect(url_for("dashboard", student_id=student_id))
    return render_template("index.html", error=error)


@app.route("/dashboard/<student_id>")
def dashboard(student_id):
    student_id = str(student_id).strip()

    if session.get("student_id") and session["student_id"] != student_id:
        if not session.get("admin_logged_in"):
            return redirect(url_for("home"))

    if not session.get("student_id") and not session.get("admin_logged_in") and not session.get("lecturer_logged_in"):
        return redirect(url_for("home"))

    student = Student.query.filter_by(student_id=student_id).first()
    pred    = Prediction.query.filter_by(student_id=student_id).first()
    if not student or not pred:
        return redirect(url_for("home"))
    if session.get("lecturer_logged_in") and not staff_can_access_student(student):
        abort(403)

    return render_template("dashboard.html", student=build_student_dict(student, pred))


def report_recommendations(student_payload):
    recs = []
    if student_payload["risk_level"] == "High Risk":
        recs.append("Schedule an academic counselling session within one week.")
    if student_payload["avg_cgpa"] < 2.5:
        recs.append("Create a subject recovery plan focused on the lowest grade-point subjects.")
    if student_payload["sem3_attendance"] < 64.3 or student_payload["total_barred"] > 0:
        recs.append("Review attendance immediately to avoid further barring from final exams.")
    return recs or ["Maintain current study habits and continue monitoring academic performance."]


def escape_pdf_text(text) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def simple_pdf_report(student_payload, recommendations) -> bytes:
    lines = [
        "EduPredict Student Report",
        f"Student ID: {student_payload['id']}",
        f"Prediction: {student_payload['prediction']} - {student_payload['risk_level']}",
        f"Risk probability: {student_payload['risk_prob']}%",
        f"Average CGPA: {student_payload['avg_cgpa']}",
        f"Overall attendance: {student_payload['overall_att']}%",
        "",
        "Recommendations",
    ]
    lines.extend([f"- {rec}" for rec in recommendations])

    content = ["BT", "/F1 12 Tf", "50 790 Td", "16 TL"]
    for line in lines[:46]:
        content.append(f"({escape_pdf_text(line)}) Tj")
        content.append("T*")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1", "replace")

    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(pdf)


@app.route("/dashboard/<student_id>/pdf")
def download_pdf(student_id):
    student_id = str(student_id).strip()
    if session.get("student_id") and session["student_id"] != student_id:
        if not session.get("admin_logged_in"):
            return redirect(url_for("home"))
    if not session.get("student_id") and not session.get("admin_logged_in"):
        return redirect(url_for("home"))

    student = Student.query.filter_by(student_id=student_id).first()
    pred = Prediction.query.filter_by(student_id=student_id).first()
    if not student or not pred:
        return redirect(url_for("home"))

    payload = build_student_dict(student, pred)
    html = render_template("dashboard_print.html",
                           student=payload,
                           recommendations=report_recommendations(payload),
                           generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"))
    recommendations = report_recommendations(payload)
    try:
        import weasyprint
        pdf = weasyprint.HTML(string=html).write_pdf()
    except Exception:
        pdf = simple_pdf_report(payload, recommendations)
    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=student_{student_id}_report.pdf"},
    )



#  AI CHAT — Gemini-powered advisor
@app.route("/api/chat", methods=["POST"])
def chat():
    # Auth: must be a logged-in student OR admin
    if not session.get("student_id") and not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    student_id  = data.get("student_id")
    user_message = data.get("message", "").strip()
    history      = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Security: students can only query their own data
    if session.get("student_id") and session.get("student_id") != student_id:
        return jsonify({"error": "Unauthorized"}), 403

    # Fetch from DB
    student     = Student.query.filter_by(student_id=student_id).first()
    prediction  = Prediction.query.filter_by(student_id=student_id).first()
    assessments = Assessment.query.filter_by(student_id=student_id).order_by(
                    Assessment.semester, Assessment.subject).all()
    attendance  = AttendanceRecord.query.filter_by(student_id=student_id).order_by(
                    AttendanceRecord.semester, AttendanceRecord.subject).all()

    if not student:
        return jsonify({"error": "Student not found"}), 404

    # Build student context string
    lines = [
        f"Student ID: {student.student_id}",
        f"Gender: {student.gender or 'N/A'}",
        f"Ethnicity: {student.ethnicity or 'N/A'}",
        f"Parental Education: {student.parental_education or 'N/A'}",
        f"Study Hours/Week: {student.study_hours or 'N/A'}",
        f"Study Method: {student.study_method or 'N/A'}",
        f"Tutoring: {student.tutoring or 'N/A'}",
    ]
    if prediction:
        lines += [
            f"Avg CGPA (Sem1-3): {prediction.avg_cgpa:.2f}",
            f"Risk Level: {prediction.risk_level}",
            f"Risk Probability: {prediction.risk_probability:.1%}",
            f"Sem1 CGPA: {prediction.sem1_cgpa:.2f}  Sem2 CGPA: {prediction.sem2_cgpa:.2f}  Sem3 CGPA: {prediction.sem3_cgpa:.2f}",
        ]
    lines.append("\n=== Assessment Records ===")
    for a in assessments:
        lines.append(
            f"Sem{a.semester} {a.subject}: Quiz={a.quiz}/20, Assn={a.assignment}/40, "
            f"Carry={a.carry_total}/60, Final={a.final_exam}/40, Total={a.total}/100, "
            f"Grade={a.grade}, GPA={a.grade_point}, {a.pass_fail or 'N/A'}"
        )
    lines.append("\n=== Attendance Records ===")
    for att in attendance:
        lines.append(
            f"Sem{att.semester} {att.subject}: Absences={att.absences}/14, "
            f"Attendance={att.attendance:.1f}%, "
            f"Barred={'YES' if att.barred else 'No'}"
        )

    student_context = "\n".join(lines)

    system_prompt = f"""You are EduPredict AI, a warm and supportive academic advisor for students at IIUM (International Islamic University Malaysia).
You have been given this student's complete academic record. Always base your advice on their ACTUAL data — never give generic responses.

{student_context}

IIUM ACADEMIC RULES:
- Each semester has 14 classes per subject. Maximum allowed absences = 5.
- More than 5 absences = BARRED from final exam = 0 marks = automatic FAIL.
- Barring threshold: attendance must stay above 64.3% (at least 9/14 classes attended).
- At-risk criteria: Avg CGPA < 2.50, OR any subject barred, OR attendance < 64.3%, OR 2+ subjects failed.
- CGPA per semester = average of 3 subject grade points. Overall = (Sem1+Sem2+Sem3)/3.
- Grade scale: 80-100=A/4.00, 75-79=A-/3.67, 70-74=B+/3.33, 65-69=B/3.00, 60-64=B-/2.67,
  55-59=C+/2.33, 50-54=C/2.00, 45-49=D/1.67, 40-44=D-/1.33, 35-39=E/1.00, 0-34=F/0.00.
- To pass a subject: need at least 24/60 carry AND 16/40 final AND 50/100 total.

YOUR BEHAVIOUR:
- Be specific: always reference the student's actual scores and subjects by name.
- Be encouraging but honest — if they are at serious risk, say so clearly but constructively.
- Give numbered, actionable steps when making recommendations.
- Keep responses under 200 words unless the student asks for more detail.
- You can do simple maths (e.g. "what final score do I need to pass?").
- Do NOT discuss other students' data or system internals."""

    try:
        if not GEMINI_ENABLED:
            return jsonify({"error": "AI chat is not configured. Add GEMINI_API_KEY to your .env file."}), 503

        model_ai = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt
        )

        # Convert history to Gemini format (uses "model" instead of "assistant")
        gemini_history = []
        for msg in history[-10:]:
            gemini_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]]
            })

        chat_session = model_ai.start_chat(history=gemini_history)
        response = chat_session.send_message(user_message)
        return jsonify({"reply": response.text})

    except Exception as e:
        err = str(e)
        print(f"  [Gemini Error] {err}")   # shows in terminal for debugging
        if "API_KEY" in err or "credentials" in err.lower() or "invalid" in err.lower() or "key" in err.lower():
            return jsonify({"error": "Invalid Gemini API key. Please check your .env file and restart the server."}), 500
        if "quota" in err.lower() or "429" in err or "rate" in err.lower():
            return jsonify({"error": "Free quota reached. Try again in a minute or tomorrow."}), 429
        if "not found" in err.lower() or "404" in err:
            return jsonify({"error": "Gemini model not available. The API may have changed."}), 500
        return jsonify({"error": f"AI error: {err}"}), 500



#  ADMIN DASHBOARD
@app.route("/admin")
@admin_required
def admin():
    stats = build_dashboard_stats(Student.query, Prediction.query)
    return render_template("admin.html", stats=stats,
                           sections=[],
                           active_section=None,
                           xgboost_available=XGBOOST_AVAILABLE,
                           is_lecturer=False,
                           dashboard_title="Admin Dashboard",
                           logout_url=url_for("admin_logout"))


@app.route("/lecturer")
@app.route("/lecturer/dashboard")
@lecturer_required
def lecturer_dashboard():
    stats = build_dashboard_stats(scoped_student_query(), scoped_prediction_query())
    return render_template("admin.html", stats=stats,
                           sections=lecturer_sections(),
                           active_section=get_active_section(),
                           xgboost_available=XGBOOST_AVAILABLE,
                           is_lecturer=True,
                           dashboard_title="Lecturer Dashboard",
                           logout_url=url_for("lecturer_logout"))



#  ADMIN — ADD / DELETE STUDENT
def _clean_text(data, key, default=""):
    value = data.get(key)
    if value is None or value == "":
        return default
    return str(value).strip()


def _clean_int(data, key, default=0):
    value = data.get(key)
    if value is None or value == "":
        return default
    return int(float(value))


def _clean_float(data, key, default=0.0):
    value = data.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _grade_letter(total):
    if total >= 80: return "A"
    if total >= 75: return "A-"
    if total >= 70: return "B+"
    if total >= 65: return "B"
    if total >= 60: return "B-"
    if total >= 55: return "C+"
    if total >= 50: return "C"
    if total >= 45: return "D"
    return "F"


def _sync_student_payload(student, pred, data):
    student.age = _clean_int(data, "Age", student.age or 20)
    student.gender = _clean_text(data, "Gender", student.gender or "Male")
    student.ethnicity = _clean_text(data, "Ethnicity", student.ethnicity or "Malay")
    student.parental_education = _clean_text(
        data, "ParentalEducation", student.parental_education or "Secondary School (SPM)")
    student.study_hours = _clean_float(data, "StudyTimeWeekly", student.study_hours or 10.0)
    student.study_method = _clean_text(data, "StudyMethod", student.study_method or "Mixed")
    student.tutoring = _clean_text(data, "Tutoring", student.tutoring or "No")
    student.extracurricular = _clean_text(data, "Extracurricular", student.extracurricular or "No")
    student.sports = _clean_text(data, "Sports", student.sports or "No")
    student.music = _clean_text(data, "Music", student.music or "No")
    student.volunteering = _clean_text(data, "Volunteering", student.volunteering or "No")

    pred.sem1_cgpa = _clean_float(data, "Sem1_CGPA", pred.sem1_cgpa or 0)
    pred.sem2_cgpa = _clean_float(data, "Sem2_CGPA", pred.sem2_cgpa or 0)
    pred.sem3_cgpa = _clean_float(data, "Sem3_CGPA", pred.sem3_cgpa or 0)
    pred.avg_cgpa = _clean_float(data, "AvgCGPA", pred.avg_cgpa or 0)
    pred.sem1_att = _clean_float(data, "Sem1_AttendancePct", pred.sem1_att or 0)
    pred.sem2_att = _clean_float(data, "Sem2_AttendancePct", pred.sem2_att or 0)
    pred.sem3_att = _clean_float(data, "Sem3_AttendancePct", pred.sem3_att or 0)
    pred.overall_att = _clean_float(data, "OverallAttendancePct", pred.overall_att or 0)
    pred.att_trend = _clean_float(data, "AttendanceTrend", pred.att_trend or 0)

    total_failed = 0
    total_barred = 0
    for sem in (1, 2, 3):
        sem_failed = 0
        sem_barred = 0
        for subject in SUBJECTS:
            prefix = f"Sem{sem}_{subject}"
            quiz = _clean_float(data, f"{prefix}_Quiz", 0)
            assignment = _clean_float(data, f"{prefix}_Assignment", 0)
            final_exam = _clean_float(data, f"{prefix}_FinalExam", 0)
            absences = _clean_int(data, f"{prefix}_Absences", 0)
            carry_total = quiz + assignment
            total = carry_total + final_exam
            barred = absences > 5
            passed = carry_total >= 24 and final_exam >= 16 and total >= 50 and not barred

            assessment = Assessment.query.filter_by(
                student_id=student.student_id, semester=sem, subject=subject).first()
            if not assessment:
                assessment = Assessment(student_id=student.student_id, semester=sem, subject=subject)
                db.session.add(assessment)
            assessment.quiz = quiz
            assessment.assignment = assignment
            assessment.carry_total = carry_total
            assessment.final_exam = final_exam
            assessment.total = total
            assessment.grade = "Bar" if barred else _grade_letter(total)
            assessment.grade_point = 0 if barred else get_grade_point(total)
            assessment.pass_fail = "Pass" if passed else "Fail"

            attendance = AttendanceRecord.query.filter_by(
                student_id=student.student_id, semester=sem, subject=subject).first()
            if not attendance:
                attendance = AttendanceRecord(student_id=student.student_id, semester=sem, subject=subject)
                db.session.add(attendance)
            attendance.absences = absences
            attendance.attendance = round(max(0, TOTAL_CLASSES_SEM - absences) / TOTAL_CLASSES_SEM * 100, 1)
            attendance.barred = barred

            sem_failed += 0 if passed else 1
            sem_barred += 1 if barred else 0
        total_failed += sem_failed
        total_barred += sem_barred
        setattr(pred, f"sem{sem}_barred", _clean_int(data, f"Sem{sem}_BarredSubjects", sem_barred))

    pred.total_failed = _clean_int(data, "TotalFailedSubjects", total_failed)
    pred.total_barred = _clean_int(data, "TotalBarredSubjects", total_barred)
    risk_prob = _clean_float(data, "RiskProbability", pred.risk_probability or 0)
    pred.risk_probability = risk_prob / 100 if risk_prob > 1 else risk_prob
    pred.prediction_label = _clean_text(data, "PredictionLabel", pred.prediction_label or "Safe")
    next_risk_level = _clean_text(data, "RiskLevel", get_risk_level(pred.risk_probability))
    maybe_notify_risk_change(student, pred, next_risk_level)
    pred.risk_level = next_risk_level


def _student_full_payload(student, pred):
    payload = {
        "student_id": student.student_id,
        "Age": student.age,
        "Gender": student.gender,
        "Ethnicity": student.ethnicity,
        "ParentalEducation": student.parental_education,
        "StudyTimeWeekly": student.study_hours,
        "StudyMethod": student.study_method,
        "Tutoring": student.tutoring,
        "Extracurricular": student.extracurricular,
        "Sports": student.sports,
        "Music": student.music,
        "Volunteering": student.volunteering,
        "Sem1_CGPA": pred.sem1_cgpa if pred else 0,
        "Sem2_CGPA": pred.sem2_cgpa if pred else 0,
        "Sem3_CGPA": pred.sem3_cgpa if pred else 0,
        "AvgCGPA": pred.avg_cgpa if pred else 0,
        "Sem1_AttendancePct": pred.sem1_att if pred else 0,
        "Sem2_AttendancePct": pred.sem2_att if pred else 0,
        "Sem3_AttendancePct": pred.sem3_att if pred else 0,
        "OverallAttendancePct": pred.overall_att if pred else 0,
        "AttendanceTrend": pred.att_trend if pred else 0,
        "Sem1_BarredSubjects": pred.sem1_barred if pred else 0,
        "Sem2_BarredSubjects": pred.sem2_barred if pred else 0,
        "Sem3_BarredSubjects": pred.sem3_barred if pred else 0,
        "TotalBarredSubjects": pred.total_barred if pred else 0,
        "TotalFailedSubjects": pred.total_failed if pred else 0,
        "RiskProbability": round((pred.risk_probability or 0) * 100, 1) if pred else 0,
        "PredictionLabel": pred.prediction_label if pred else "Safe",
        "RiskLevel": pred.risk_level if pred else "Low Risk",
    }
    for sem in (1, 2, 3):
        for subject in SUBJECTS:
            prefix = f"Sem{sem}_{subject}"
            assessment = Assessment.query.filter_by(
                student_id=student.student_id, semester=sem, subject=subject).first()
            attendance = AttendanceRecord.query.filter_by(
                student_id=student.student_id, semester=sem, subject=subject).first()
            payload[f"{prefix}_Quiz"] = assessment.quiz if assessment else 0
            payload[f"{prefix}_Assignment"] = assessment.assignment if assessment else 0
            payload[f"{prefix}_FinalExam"] = assessment.final_exam if assessment else 0
            payload[f"{prefix}_Absences"] = attendance.absences if attendance else 0
    return payload


@app.route("/api/students/add", methods=["POST"])
@login_required
def api_add_student():
    try:
        data       = request.get_json() or {}
        student_id = str(data.get("student_id", "")).strip()
        if not student_id:
            return jsonify({"success": False, "error": "Student ID is required."}), 400
        if Student.query.filter_by(student_id=student_id).first():
            return jsonify({"success": False,
                            "error": f"Student ID {student_id} already exists."}), 400
        active_section = get_active_section()
        if session.get("lecturer_logged_in") and not active_section:
            return jsonify({"success": False, "error": "Create or upload a class section before adding students."}), 400
        s = Student(student_id=student_id,
                    section_id=active_section.id if active_section else None,
                    registration_status="preloaded")
        p = Prediction(
            student_id=student_id,
            sem1_cgpa=0, sem2_cgpa=0, sem3_cgpa=0, avg_cgpa=0,
            overall_att=0, att_trend=0, risk_probability=0.0,
            prediction_label="Safe", risk_level="Low Risk",
        )
        db.session.add(s)
        db.session.add(p)
        _sync_student_payload(s, p, data)
        db.session.commit()
        return jsonify({"success": True, "message": f"Student {student_id} added."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/students/import", methods=["POST"])
@login_required
def api_import_students():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify({"success": False, "error": "Please choose a CSV or Excel file."}), 400

    filename = upload.filename.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(upload)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(upload)
        else:
            return jsonify({"success": False, "error": "Only .csv, .xlsx, or .xls files are supported."}), 400

        imported = skipped = 0
        active_section = get_active_section()
        if session.get("lecturer_logged_in") and not active_section:
            return jsonify({"success": False, "error": "Create or upload a class section before importing students."}), 400
        for _, row in df.iterrows():
            row_dict = {str(k).strip(): v for k, v in row.to_dict().items()}
            student_id = str(
                row_dict.get("student_id")
                or row_dict.get("StudentID")
                or row_dict.get("Student_ID")
                or row_dict.get("id")
                or ""
            ).strip()
            if not student_id or student_id.lower() == "nan":
                skipped += 1
                continue
            if Student.query.filter_by(student_id=student_id).first():
                skipped += 1
                continue

            def val(*names, default=None):
                for name in names:
                    raw = row_dict.get(name)
                    if pd.notna(raw) and str(raw).strip():
                        return raw
                return default

            s = Student(
                student_id=student_id,
                section_id=active_section.id if active_section else None,
                registration_status="preloaded",
                age=int(float(val("Age", "age", default=20))),
                gender=str(val("Gender", "gender", default="Male")),
                ethnicity=str(val("Ethnicity", "ethnicity", default="Malay")),
                parental_education=str(val("ParentalEducation", "parental_education", default="Secondary School (SPM)")),
                study_hours=float(val("StudyTimeWeekly", "study_hours", default=10.0)),
                study_method=str(val("StudyMethod", "study_method", default="Mixed")),
                tutoring=str(val("Tutoring", "tutoring", default="No")),
                extracurricular=str(val("Extracurricular", "extracurricular", default="No")),
                sports=str(val("Sports", "sports", default="No")),
                music=str(val("Music", "music", default="No")),
                volunteering=str(val("Volunteering", "volunteering", default="No")),
            )
            p = Prediction(
                student_id=student_id,
                sem1_cgpa=float(val("Sem1_CGPA", "sem1_cgpa", default=0)),
                sem2_cgpa=float(val("Sem2_CGPA", "sem2_cgpa", default=0)),
                sem3_cgpa=float(val("Sem3_CGPA", "sem3_cgpa", default=0)),
                avg_cgpa=float(val("AvgCGPA", "avg_cgpa", default=0)),
                sem1_att=float(val("Sem1_AttendancePct", "sem1_att", default=0)),
                sem2_att=float(val("Sem2_AttendancePct", "sem2_att", default=0)),
                sem3_att=float(val("Sem3_AttendancePct", "sem3_att", default=0)),
                overall_att=float(val("OverallAttendancePct", "overall_att", default=0)),
                att_trend=float(val("AttendanceTrend", "att_trend", default=0)),
                risk_probability=float(val("risk_probability", "RiskProbability", default=0)),
                prediction_label=str(val("prediction_label", "Prediction", default="Safe")),
                risk_level=str(val("risk_level", "RiskLevel", default="Low Risk")),
            )
            db.session.add(s)
            db.session.add(p)
            imported += 1

        db.session.commit()
        return jsonify({
            "success": True,
            "message": f"Imported {imported} student(s). Skipped {skipped}.",
            "imported": imported,
            "skipped": skipped,
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


def _row_value(row_dict, *names, default=None):
    for name in names:
        raw = row_dict.get(name)
        if pd.notna(raw) and str(raw).strip():
            return raw
    return default


@app.route("/lecturer/switch-section/<int:section_id>")
@lecturer_required
def switch_section(section_id):
    section = ClassSection.query.get_or_404(section_id)
    if section.lecturer_id != current_lecturer_id():
        abort(403)
    session["active_section_id"] = section.id
    return redirect(url_for("lecturer_dashboard"))


@app.route("/api/sections")
@lecturer_required
def api_sections():
    rows = []
    for section in lecturer_sections(include_archived=True):
        q = (db.session.query(Student, Prediction)
             .join(Prediction, Student.student_id == Prediction.student_id)
             .filter(Student.section_id == section.id))
        total = Student.query.filter_by(section_id=section.id).count()
        at_risk = q.filter(Prediction.prediction_label == "At-Risk").count()
        avg_cgpa = q.with_entities(db.func.avg(Prediction.avg_cgpa)).scalar() or 0
        student_rows = q.order_by(Prediction.risk_probability.desc()).all()
        rows.append({
            "id": section.id,
            "name": section.section_name,
            "semester": section.current_semester,
            "sem_depth": section.sem_depth,
            "students": total,
            "at_risk": at_risk,
            "avg_cgpa": round(float(avg_cgpa), 2),
            "archived": section.is_archived,
            "active": section.id == session.get("active_section_id"),
            "model": "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting fallback",
            "student_rows": [{
                "id": s.student_id,
                "gender": s.gender or "-",
                "avg_cgpa": round(float(p.avg_cgpa or 0), 2),
                "risk_level": p.risk_level,
                "prediction": p.prediction_label,
                "risk_prob": round(float(p.risk_probability or 0) * 100, 1),
            } for s, p in student_rows],
        })
    return jsonify({"sections": rows})


@app.route("/api/sections/template")
@lecturer_required
def section_template():
    sem_depth = int(request.args.get("semester", 3))
    cols = expected_template_columns(max(1, min(3, sem_depth)))
    sample = {c: "" for c in cols}
    sample.update({
        "StudentID": "S1001", "Age": 20, "Gender": "Female",
        "Ethnicity": "Malay", "ParentalEducation": "Secondary School (SPM)",
        "StudyTimeWeekly": 10, "StudyMethod": "Mixed",
        "Tutoring": "No", "Extracurricular": "Yes",
        "Sports": "No", "Music": "No", "Volunteering": "Yes",
        "PredictionLabel": "Safe", "RiskLevel": "Low Risk", "RiskProbability": 0.18,
    })
    csv_text = pd.DataFrame([sample], columns=cols).to_csv(index=False)
    return Response(csv_text, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=section_sem{sem_depth}_template.csv"})


@app.route("/api/sections/upload", methods=["POST"])
@lecturer_required
def upload_section():
    upload = request.files.get("csv_file")
    section_name = request.form.get("section_name", "").strip()
    current_semester = int(request.form.get("current_semester", 3))
    if not upload or not upload.filename or not section_name:
        return jsonify({"success": False, "error": "Section name and CSV file are required."}), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Section onboarding currently accepts CSV files only."}), 400
    try:
        df = pd.read_csv(upload)
        sem_depth = detect_semester_depth(df.columns)
        required = {"StudentID", "PredictionLabel"}
        missing = required - set(df.columns)
        if missing:
            return jsonify({"success": False, "error": f"Missing columns: {', '.join(sorted(missing))}"}), 400

        lecturer_id = current_lecturer_id()
        section = ClassSection.query.filter_by(
            lecturer_id=lecturer_id, section_name=section_name).first()
        if not section:
            section = ClassSection(
                lecturer_id=lecturer_id,
                section_name=section_name,
                current_semester=current_semester,
                sem_depth=sem_depth,
            )
            db.session.add(section)
            db.session.flush()
        else:
            section.current_semester = current_semester

        safe_name = secure_filename(f"{section_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv")
        csv_path = os.path.join(UPLOAD_FOLDER, safe_name)
        df.to_csv(csv_path, index=False)
        section.dataset_path = csv_path
        model_name = train_model_for_section(csv_path, section)

        imported = updated = skipped = 0
        for _, row in df.iterrows():
            row_dict = {str(k).strip(): v for k, v in row.to_dict().items()}
            student_id = str(_row_value(row_dict, "StudentID", "student_id", "Student_ID", "id", default="")).strip()
            if not student_id or student_id.lower() == "nan":
                skipped += 1
                continue
            student = Student.query.filter_by(student_id=student_id).first()
            is_new = student is None
            if is_new:
                student = Student(student_id=student_id, registration_status="preloaded")
                db.session.add(student)
            student.section_id = section.id
            pred = Prediction.query.filter_by(student_id=student_id).first()
            if not pred:
                pred = Prediction(student_id=student_id, risk_probability=0,
                                  prediction_label="Safe", risk_level="Low Risk")
                db.session.add(pred)
            _sync_student_payload(student, pred, row_dict)
            imported += 1 if is_new else 0
            updated += 0 if is_new else 1

        if session.get("active_section_id") is None:
            session["active_section_id"] = section.id
        db.session.commit()
        return jsonify({
            "success": True,
            "message": f"{section.section_name}: {imported} added, {updated} updated, {skipped} skipped. Trained {model_name} sem-depth {section.sem_depth} model.",
            "students_loaded": imported + updated,
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/notifications")
@lecturer_required
def api_notifications():
    section = get_active_section()
    q = RiskNotification.query
    if section:
        q = q.filter_by(section_id=section.id)
    rows = q.order_by(RiskNotification.created_at.desc()).limit(25).all()
    return jsonify({"notifications": [{
        "student_id": r.student_id,
        "old": r.old_risk_level,
        "new": r.new_risk_level,
        "recipient": r.recipient or "no email",
        "sent": r.sent,
        "created_at": r.created_at.strftime("%Y-%m-%d %H:%M"),
    } for r in rows]})


@app.route("/api/students/<student_id>/full")
@login_required
def api_student_full(student_id):
    student = Student.query.filter_by(student_id=str(student_id)).first()
    if not student:
        return jsonify({"success": False, "error": "Student not found."}), 404
    if not staff_can_access_student(student):
        return jsonify({"success": False, "error": "Not allowed for this student."}), 403
    pred = Prediction.query.filter_by(student_id=student.student_id).first()
    return jsonify({"success": True, "student": _student_full_payload(student, pred)})


@app.route("/api/students/update/<student_id>", methods=["POST"])
@login_required
def api_update_student(student_id):
    try:
        student = Student.query.filter_by(student_id=str(student_id)).first()
        if not student:
            return jsonify({"success": False, "error": "Student not found."}), 404
        if not staff_can_access_student(student):
            return jsonify({"success": False, "error": "Not allowed for this student."}), 403
        data = request.get_json() or {}
        original_id = student.student_id
        new_id = str(data.get("student_id", student.student_id)).strip()
        if new_id and new_id != student.student_id:
            if Student.query.filter_by(student_id=new_id).first():
                return jsonify({"success": False,
                                "error": f"Student ID {new_id} already exists."}), 400
            old_id = student.student_id
            for assessment in Assessment.query.filter_by(student_id=old_id).all():
                assessment.student_id = new_id
            for attendance in AttendanceRecord.query.filter_by(student_id=old_id).all():
                attendance.student_id = new_id
            student.student_id = new_id
        pred = (Prediction.query.filter_by(student_id=student.student_id).first()
                or Prediction.query.filter_by(student_id=original_id).first())
        if not pred:
            pred = Prediction(
                student_id=student.student_id,
                sem1_cgpa=0, sem2_cgpa=0, sem3_cgpa=0, avg_cgpa=0,
                overall_att=0, att_trend=0, risk_probability=0.0,
                prediction_label="Safe", risk_level="Low Risk",
            )
            db.session.add(pred)
        else:
            pred.student_id = student.student_id
        _sync_student_payload(student, pred, data)
        db.session.commit()
        return jsonify({"success": True, "message": f"Student {student.student_id} updated."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/students/delete/<student_id>", methods=["DELETE"])
@login_required
def api_delete_student(student_id):
    try:
        s = Student.query.filter_by(student_id=str(student_id)).first()
        if not s:
            return jsonify({"success": False,
                            "error": f"Student ID {student_id} not found."}), 404
        if not staff_can_access_student(s):
            return jsonify({"success": False, "error": "Not allowed for this student."}), 403
        db.session.delete(s)
        db.session.commit()
        return jsonify({"success": True, "message": f"Student {student_id} deleted."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500



#  JSON API — admin dashboard charts
@app.route("/api/risk-split")
@login_required
def api_risk_split():
    q = scoped_prediction_query()
    return jsonify({
        "Safe":    q.filter(Prediction.prediction_label == "Safe").count(),
        "At-Risk": q.filter(Prediction.prediction_label == "At-Risk").count(),
    })


@app.route("/api/risk-levels")
@login_required
def api_risk_levels():
    q = scoped_prediction_query()
    return jsonify({
        "High Risk":   q.filter(Prediction.risk_level == "High Risk").count(),
        "Medium Risk": q.filter(Prediction.risk_level == "Medium Risk").count(),
        "Low Risk":    q.filter(Prediction.risk_level == "Low Risk").count(),
    })


@app.route("/api/ethnicity-risk")
@login_required
def api_ethnicity_risk():
    rows = (db.session.query(
                Student.ethnicity,
                db.func.count(Prediction.student_id).label("total"),
                db.func.sum(db.case(
                    (Prediction.prediction_label == "At-Risk", 1), else_=0
                )).label("at_risk"))
            .join(Prediction, Student.student_id == Prediction.student_id)
            .filter(lecturer_section_filter())
            .group_by(Student.ethnicity).all())
    result = [{"ethnicity": r.ethnicity,
               "at_risk_pct": round(r.at_risk / r.total * 100, 1) if r.total else 0}
              for r in rows]
    return jsonify(sorted(result, key=lambda x: x["at_risk_pct"], reverse=True))


@app.route("/api/gender-risk")
@login_required
def api_gender_risk():
    rows = (db.session.query(
                Student.gender,
                db.func.count(Prediction.student_id).label("total"),
                db.func.sum(db.case(
                    (Prediction.prediction_label == "At-Risk", 1), else_=0
                )).label("at_risk"))
            .join(Prediction, Student.student_id == Prediction.student_id)
            .filter(lecturer_section_filter())
            .group_by(Student.gender).all())
    return jsonify([{"gender": r.gender,
                     "at_risk_pct": round(r.at_risk / r.total * 100, 1) if r.total else 0}
                    for r in rows])


@app.route("/api/cgpa-trend")
@login_required
def api_cgpa_trend():
    avg = scoped_prediction_query().with_entities(
        db.func.avg(Prediction.sem1_cgpa),
        db.func.avg(Prediction.sem2_cgpa),
        db.func.avg(Prediction.sem3_cgpa),
    ).first()
    return jsonify({
        "Sem 1": round(float(avg[0] or 0), 3),
        "Sem 2": round(float(avg[1] or 0), 3),
        "Sem 3": round(float(avg[2] or 0), 3),
    })


@app.route("/api/study-method")
@login_required
def api_study_method():
    rows = (db.session.query(
                Student.study_method,
                db.func.avg(Prediction.avg_cgpa).label("avg_cgpa"))
            .join(Prediction, Student.student_id == Prediction.student_id)
            .filter(lecturer_section_filter())
            .group_by(Student.study_method).all())
    return jsonify(sorted(
        [{"method": r.study_method, "avg_cgpa": round(float(r.avg_cgpa or 0), 3)}
         for r in rows],
        key=lambda x: x["avg_cgpa"], reverse=True))


@app.route("/api/attendance-risk")
@login_required
def api_attendance_risk():
    bins = [(0,50,"0-50"),(51,60,"51-60"),(61,70,"61-70"),
            (71,80,"71-80"),(81,90,"81-90"),(91,100,"91-100")]
    result = []
    for lo, hi, label in bins:
        q = scoped_prediction_query()
        total   = q.filter(
            Prediction.overall_att.between(lo, hi)).count()
        at_risk = q.filter(
            Prediction.overall_att.between(lo, hi),
            Prediction.prediction_label == "At-Risk").count()
        result.append({"range": label,
                       "at_risk_pct": round(at_risk / total * 100, 1) if total else 0})
    return jsonify(result)


@app.route("/api/parental-education")
@login_required
def api_parental_education():
    rows = (db.session.query(
                Student.parental_education,
                db.func.avg(Prediction.avg_cgpa).label("avg_cgpa"))
            .join(Prediction, Student.student_id == Prediction.student_id)
            .filter(lecturer_section_filter())
            .group_by(Student.parental_education).all())
    return jsonify(sorted(
        [{"education": r.parental_education,
          "avg_cgpa": round(float(r.avg_cgpa or 0), 3)} for r in rows],
        key=lambda x: x["avg_cgpa"], reverse=True))


@app.route("/api/students")
@login_required
def api_students():
    search      = request.args.get("search", "").strip().lower()
    risk_filter = request.args.get("risk", "all")
    page        = int(request.args.get("page", 1))
    per_page    = 15

    q = (db.session.query(Student, Prediction)
         .join(Prediction, Student.student_id == Prediction.student_id))
    if current_lecturer_id():
        active_section = get_active_section()
        if active_section:
            q = q.filter(Student.section_id == active_section.id)
        else:
            q = q.filter(false())
    if risk_filter != "all":
        q = q.filter(Prediction.risk_level == risk_filter)
    if search:
        q = q.filter(db.or_(
            Student.student_id.ilike(f"%{search}%"),
            Student.gender.ilike(f"%{search}%"),
            Student.ethnicity.ilike(f"%{search}%"),
        ))
    total_rows = q.count()
    rows = (q.order_by(Prediction.risk_probability.desc())
            .offset((page - 1) * per_page).limit(per_page).all())
    records = [
        {"id": s.student_id, "gender": s.gender, "ethnicity": s.ethnicity,
         "attendance": round(float(p.overall_att or 0), 1),
         "avg_cgpa":   round(float(p.avg_cgpa    or 0), 2),
         "prediction": p.prediction_label,
         "risk_level": p.risk_level,
         "risk_prob":  round(float(p.risk_probability) * 100, 1)}
        for s, p in rows
    ]
    return jsonify({"total": total_rows, "page": page,
                    "per_page": per_page, "students": records})


@app.route("/api/live-stats")
@login_required
def api_live_stats():
    total   = LivePrediction.query.count()
    at_risk = LivePrediction.query.filter_by(prediction="At-Risk").count()
    safe    = LivePrediction.query.filter_by(prediction="Safe").count()
    avg_q   = db.session.query(
        db.func.avg(LivePrediction.risk_probability)).scalar()
    recent  = (LivePrediction.query
               .order_by(LivePrediction.timestamp.desc()).limit(10).all())
    return jsonify({
        "total":         total,
        "at_risk":       at_risk,
        "safe":          safe,
        "avg_risk_prob": round(float(avg_q or 0) * 100, 1),
        "recent": [
            {"timestamp":        r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
             "prediction":       r.prediction,
             "risk_level":       r.risk_level,
             "risk_probability": round(float(r.risk_probability) * 100, 1),
             "avg_cgpa":         round(float(r.avg_cgpa or 0), 2)}
            for r in recent
        ],
    })



#  PREDICT — log to live_predictions table
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    active_section = get_active_section() if session.get("lecturer_logged_in") else None
    confidence = model_confidence(active_section.sem_depth if active_section else 3)
    if request.method == "GET":
        return render_template("predict.html", confidence=confidence,
                               active_section=active_section,
                               xgboost_available=XGBOOST_AVAILABLE)
    try:
        f = request.form
        def flt(key, default=0.0): return float(f.get(key, default) or default)
        def ing(key, default=0):   return int(float(f.get(key, default) or default))

        raw = {
            # Attendance
            "att3":  flt("att3"),   "abs3":  ing("abs3"),
            "sem3":  flt("sem3"),
            # Per-subject carry marks
            "math_quiz": flt("math_quiz"), "math_assn": flt("math_assn"), "math_final": flt("math_final"),
            "sci_quiz":  flt("sci_quiz"),  "sci_assn":  flt("sci_assn"),  "sci_final":  flt("sci_final"),
            "eng_quiz":  flt("eng_quiz"),  "eng_assn":  flt("eng_assn"),  "eng_final":  flt("eng_final"),
            # Per-subject absences (optional)
            "math_abs3": flt("math_abs3", ing("abs3") / 3),
            "sci_abs3":  flt("sci_abs3",  ing("abs3") / 3),
            "eng_abs3":  flt("eng_abs3",  ing("abs3") / 3),
            # Student profile
            "study_hours":     flt("study_hours"),
            "gender":          f.get("gender",         "Male"),
            "ethnicity":       f.get("ethnicity",      "Malay"),
            "education":       f.get("education",      "Secondary School (SPM)"),
            "method":          f.get("method",         "Mixed"),
            "tutoring":        f.get("tutoring",       "No"),
            "extracurricular": f.get("extracurricular","No"),
            "sports":          f.get("sports",         "No"),
            "music":           f.get("music",          "No"),
            "volunteering":    f.get("volunteering",   "No"),
        }

        avg_cgpa = round(raw["sem3"], 4)   # single sem — use Sem3 CGPA as proxy

        # Build features and run the most specific compatible model.
        input_df     = build_predict_features(raw)
        selected_model = model_sem3
        selected_features = FEATURES_SEM3
        model_note = "Global Sem 3 model"
        if active_section and active_section.model_path and active_section.features_path:
            try:
                section_features = joblib.load(active_section.features_path)
                if set(section_features).issubset(set(input_df.columns)):
                    selected_model = joblib.load(active_section.model_path)
                    selected_features = section_features
                    model_note = f"{active_section.section_name} section model"
            except Exception:
                model_note = "Global Sem 3 model"
        scored_df = input_df.reindex(columns=selected_features, fill_value=0)
        prediction   = selected_model.predict(scored_df)[0]
        risk_prob    = round(float(selected_model.predict_proba(scored_df)[0][1]), 4)
        risk_lbl     = get_risk_level(risk_prob)
        result_label = "At-Risk" if prediction == 1 else "Safe"
        risk_factors = manual_prediction_factors(raw, risk_prob)

        # Log to live_predictions table
        db.session.add(LivePrediction(
            attendance=raw["att3"],
            quiz=round((raw["math_quiz"]+raw["sci_quiz"]+raw["eng_quiz"])/3, 1),
            assignment=round((raw["math_assn"]+raw["sci_assn"]+raw["eng_assn"])/3, 1),
            math=round(raw["math_quiz"]+raw["math_assn"]+raw["math_final"], 1),
            science=round(raw["sci_quiz"]+raw["sci_assn"]+raw["sci_final"], 1),
            english=round(raw["eng_quiz"]+raw["eng_assn"]+raw["eng_final"], 1),
            sem1_cgpa=raw["sem3"], sem2_cgpa=raw["sem3"], sem3_cgpa=raw["sem3"],
            avg_cgpa=avg_cgpa, study_hours=raw["study_hours"],
            absences=raw["abs3"], gender=raw["gender"],
            ethnicity=raw["ethnicity"], education=raw["education"],
            study_method=raw["method"], tutoring=raw["tutoring"],
            extracurricular=raw["extracurricular"], sports=raw["sports"],
            music=raw["music"], volunteering=raw["volunteering"],
            prediction=result_label, risk_probability=risk_prob, risk_level=risk_lbl,
        ))
        db.session.commit()

        return render_template("predict.html",
                               prediction_text=result_label,
                               risk_prob=f"{risk_prob*100:.1f}%",
                               risk_level=risk_lbl,
                               avg_cgpa=round(avg_cgpa, 2),
                               confidence=confidence,
                               active_section=active_section,
                               model_note=model_note,
                               risk_factors=risk_factors,
                               xgboost_available=XGBOOST_AVAILABLE)
    except Exception as e:
        db.session.rollback()
        return render_template("predict.html", errors=[str(e)],
                               confidence=confidence,
                               active_section=active_section,
                               xgboost_available=XGBOOST_AVAILABLE)



#  ENTRY POINT
if os.environ.get("AUTO_INIT_DB", "true").lower() == "true":
    try:
        init_db()
    except Exception as e:
        print(f"  Database initialization skipped: {e}")

if __name__ == "__main__":
    init_db()
    app.run(debug=os.environ.get("FLASK_DEBUG", "true").lower() == "true")
