"""
EduPredict — database.py
════════════════════════════════════════════════════════════════════
SQLAlchemy models for EduPredict.
This file defines database tables ONLY — no Flask routes here.

Tables:
  students          — student profile and demographics
  assessments       — per-subject per-semester marks, grades, GPA points
  attendance_records— per-subject per-semester absences + barring status
  predictions       — ML model output (risk label, probability, CGPA)
  live_predictions  — web form submissions logged in real-time
  admin_users       — bcrypt-hashed admin credentials
════════════════════════════════════════════════════════════════════
"""

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt      import Bcrypt
from datetime          import datetime

db     = SQLAlchemy()
bcrypt = Bcrypt()



#  STUDENTS
class Student(db.Model):
    __tablename__ = "students"

    student_id         = db.Column(db.String(20),  primary_key=True)
    section_id         = db.Column(db.Integer, db.ForeignKey("class_sections.id"))
    email              = db.Column(db.String(120), unique=True)
    password_hash      = db.Column(db.String(255))
    registration_status= db.Column(db.String(20), default="pending")
    age                = db.Column(db.Integer)
    gender             = db.Column(db.String(10))
    ethnicity          = db.Column(db.String(50))
    parental_education = db.Column(db.String(60))
    study_hours        = db.Column(db.Float)
    study_method       = db.Column(db.String(30))
    tutoring           = db.Column(db.String(5))
    extracurricular    = db.Column(db.String(5))
    sports             = db.Column(db.String(5))
    music              = db.Column(db.String(5))
    volunteering       = db.Column(db.String(5))
    created_at         = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at         = db.Column(db.DateTime, default=datetime.utcnow,
                                   onupdate=datetime.utcnow)

    assessments        = db.relationship("Assessment",       back_populates="student",
                                         cascade="all, delete-orphan")
    attendance_records = db.relationship("AttendanceRecord", back_populates="student",
                                         cascade="all, delete-orphan")
    prediction         = db.relationship("Prediction",       back_populates="student",
                                         uselist=False, cascade="all, delete-orphan")

    def set_password(self, password: str):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        return bcrypt.check_password_hash(self.password_hash, password)


class Lecturer(db.Model):
    __tablename__ = "lecturers"

    id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username      = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name     = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(120))
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    sections = db.relationship("ClassSection", back_populates="lecturer",
                               cascade="all, delete-orphan")

    def set_password(self, password: str):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password: str) -> bool:
        return bcrypt.check_password_hash(self.password_hash, password)


class ClassSection(db.Model):
    __tablename__ = "class_sections"

    id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    lecturer_id      = db.Column(db.Integer, db.ForeignKey("lecturers.id"), nullable=False)
    section_name     = db.Column(db.String(50), nullable=False)
    current_semester = db.Column(db.Integer, default=3)
    sem_depth        = db.Column(db.Integer, default=3)
    model_path       = db.Column(db.String(255))
    features_path    = db.Column(db.String(255))
    dataset_path     = db.Column(db.String(255))
    is_archived      = db.Column(db.Boolean, default=False)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at       = db.Column(db.DateTime, default=datetime.utcnow,
                                 onupdate=datetime.utcnow)

    lecturer = db.relationship("Lecturer", back_populates="sections")
    students = db.relationship("Student", backref="section")

    __table_args__ = (
        db.UniqueConstraint("lecturer_id", "section_name",
                            name="uq_lecturer_section_name"),
    )



#  ASSESSMENTS  (per subject per semester)
class Assessment(db.Model):
    __tablename__ = "assessments"

    id          = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_id  = db.Column(db.String(20), db.ForeignKey("students.student_id",
                            ondelete="CASCADE"), nullable=False)
    semester    = db.Column(db.Integer,    nullable=False)   # 1, 2, or 3
    subject     = db.Column(db.String(20), nullable=False)   # Math / Science / English

    quiz        = db.Column(db.Float)    # /20
    assignment  = db.Column(db.Float)    # /40
    carry_total = db.Column(db.Float)    # /60
    final_exam  = db.Column(db.Float)    # /40  (0 if barred)
    total       = db.Column(db.Float)    # /100
    grade       = db.Column(db.String(5))
    grade_point = db.Column(db.Float)
    pass_fail   = db.Column(db.String(5))

    student = db.relationship("Student", back_populates="assessments")

    __table_args__ = (
        db.UniqueConstraint("student_id", "semester", "subject",
                            name="uq_assessment_student_sem_subj"),
    )



#  ATTENDANCE  (per subject per semester — IIUM rules)
class AttendanceRecord(db.Model):
    __tablename__ = "attendance_records"

    id          = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_id  = db.Column(db.String(20), db.ForeignKey("students.student_id",
                            ondelete="CASCADE"), nullable=False)
    semester    = db.Column(db.Integer,    nullable=False)
    subject     = db.Column(db.String(20), nullable=False)

    absences    = db.Column(db.Integer)          # 0–14
    attendance  = db.Column(db.Float)            # percentage 0–100
    barred      = db.Column(db.Boolean, default=False)

    student = db.relationship("Student", back_populates="attendance_records")

    __table_args__ = (
        db.UniqueConstraint("student_id", "semester", "subject",
                            name="uq_attend_student_sem_subj"),
    )



#  PREDICTIONS  (ML model output — one row per student)
class Prediction(db.Model):
    __tablename__ = "predictions"

    id               = db.Column(db.Integer,     primary_key=True, autoincrement=True)
    student_id       = db.Column(db.String(20),  db.ForeignKey("students.student_id",
                                 ondelete="CASCADE"), unique=True, nullable=False)

    sem1_cgpa        = db.Column(db.Float)
    sem2_cgpa        = db.Column(db.Float)
    sem3_cgpa        = db.Column(db.Float)
    avg_cgpa         = db.Column(db.Float)

    sem1_att         = db.Column(db.Float)
    sem2_att         = db.Column(db.Float)
    sem3_att         = db.Column(db.Float)
    overall_att      = db.Column(db.Float)
    att_trend        = db.Column(db.Float)

    sem1_barred      = db.Column(db.Integer, default=0)
    sem2_barred      = db.Column(db.Integer, default=0)
    sem3_barred      = db.Column(db.Integer, default=0)
    total_barred     = db.Column(db.Integer, default=0)
    total_failed     = db.Column(db.Integer, default=0)

    risk_probability = db.Column(db.Float,      nullable=False)
    prediction_label = db.Column(db.String(10), nullable=False)
    risk_level       = db.Column(db.String(15), nullable=False)

    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at       = db.Column(db.DateTime, default=datetime.utcnow,
                                 onupdate=datetime.utcnow)

    student = db.relationship("Student", back_populates="prediction")



#  LIVE PREDICTIONS  (web form submissions)
class LivePrediction(db.Model):
    __tablename__ = "live_predictions"

    id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp        = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    attendance       = db.Column(db.Float)
    quiz             = db.Column(db.Float)
    assignment       = db.Column(db.Float)
    math             = db.Column(db.Float)
    science          = db.Column(db.Float)
    english          = db.Column(db.Float)
    sem1_cgpa        = db.Column(db.Float)
    sem2_cgpa        = db.Column(db.Float)
    sem3_cgpa        = db.Column(db.Float)
    avg_cgpa         = db.Column(db.Float)
    study_hours      = db.Column(db.Float)
    absences         = db.Column(db.Integer)
    gender           = db.Column(db.String(10))
    ethnicity        = db.Column(db.String(50))
    education        = db.Column(db.String(60))
    study_method     = db.Column(db.String(30))
    tutoring         = db.Column(db.String(5))
    extracurricular  = db.Column(db.String(5))
    sports           = db.Column(db.String(5))
    music            = db.Column(db.String(5))
    volunteering     = db.Column(db.String(5))

    prediction       = db.Column(db.String(10))
    risk_probability = db.Column(db.Float)
    risk_level       = db.Column(db.String(15))

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if self.timestamp:
            d["timestamp"] = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return d



#  ADMIN USERS  (hashed passwords)
class RiskNotification(db.Model):
    __tablename__ = "risk_notifications"

    id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_id      = db.Column(db.String(20), db.ForeignKey("students.student_id",
                                ondelete="CASCADE"), nullable=False, index=True)
    section_id      = db.Column(db.Integer, db.ForeignKey("class_sections.id"))
    old_risk_level  = db.Column(db.String(15))
    new_risk_level  = db.Column(db.String(15), nullable=False)
    channel         = db.Column(db.String(20), default="email")
    recipient       = db.Column(db.String(120))
    message         = db.Column(db.Text)
    sent            = db.Column(db.Boolean, default=False)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    student = db.relationship("Student")
    section = db.relationship("ClassSection")


class AdminUser(db.Model):
    __tablename__ = "admin_users"

    id            = db.Column(db.Integer,     primary_key=True, autoincrement=True)
    username      = db.Column(db.String(50),  unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    def set_password(self, password: str):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password: str) -> bool:
        return bcrypt.check_password_hash(self.password_hash, password)
