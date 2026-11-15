"""
EduPredict — assign_sections.py
════════════════════════════════════════════════════════════════════
Spreads the demo students across class sections shared between the
lecturers, so the demo looks like a real faculty.

Each section gets a similar mix of High / Medium / Low risk students,
and the split is the same every time for the same --seed.

Usage:
  python assign_sections.py --dry-run                 # show the plan only
  python assign_sections.py                           # 15 lecturers × 4 sections
  python assign_sections.py --lecturers 10 --per-lecturer 3
  python assign_sections.py --password "Lecturer@2026"

Missing lecturers are created with made-up names (existing lecturers keep
their passwords). New lecturers get --password, or a generated one that is
printed at the end; the admin can reset it from the dashboard.

Students moved:
  - everyone in "Default Section"
  - everyone in sections this script made before (they use the global
    Sem 3 model), so re-running with new numbers rebuilds the split
Sections a lecturer created by uploading a CSV are never touched.

Works with whichever DATABASE_URL is set in your .env file.
════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import random
import secrets

from app      import app
from database import db, Lecturer, ClassSection, Student, Prediction, RiskNotification

RISK_LEVELS = ("High Risk", "Medium Risk", "Low Risk")
GLOBAL_MODEL = "model_sem3.pkl"
DEMO_EMAIL_DOMAIN = "edupredict.test"

# Made-up faculty names for demo lecturers
LECTURER_POOL = [
    ("Dr. Ahmad Faizal bin Hassan", "ahmadfaizal"),
    ("Dr. Siti Rohani binti Ismail", "sitirohani"),
    ("Assoc. Prof. Dr. Tan Chee Wei", "tancheewei"),
    ("Dr. Kavitha a/p Ramasamy", "kavitha"),
    ("Dr. Mohd Hafiz bin Abdullah", "mohdhafiz"),
    ("Dr. Lim Mei Ling", "limmeiling"),
    ("Ts. Dr. Nurul Huda binti Rahman", "nurulhuda"),
    ("Dr. Jerry anak Jimbun", "jerryjimbun"),
    ("Dr. Aisyah binti Kamarudin", "aisyahk"),
    ("Assoc. Prof. Dr. Suresh a/l Krishnan", "sureshk"),
    ("Dr. Wong Kar Mun", "wongkarmun"),
    ("Dr. Zulkifli bin Osman", "zulkifli"),
    ("Dr. Farah Nadia binti Yusof", "farahnadia"),
    ("Dr. Ong Boon Hock", "ongboonhock"),
    ("Dr. Juliana Majimbun", "julianam"),
    ("Dr. Amirul Hakim bin Salleh", "amirulhakim"),
    ("Dr. Priya a/p Subramaniam", "priyas"),
    ("Dr. Chong Wei Jie", "chongweijie"),
    ("Dr. Hani Sabrina binti Zainal", "hanisabrina"),
    ("Dr. Iqbal Hakim bin Razak", "iqbalhakim"),
]


def is_demo_section(section: ClassSection) -> bool:
    """Default Section or one this script made (both use the global Sem 3 model, not a trained upload)."""
    if section.section_name == "Default Section":
        return True
    return (section.section_name.startswith("Section ")
            and os.path.basename(section.model_path or "") == GLOBAL_MODEL)


def ensure_lecturers(target: int, password: str, dry_run: bool):
    """Return `target` active lecturers, creating made-up ones if there are too few."""
    lecturers = Lecturer.query.filter_by(is_active=True).order_by(Lecturer.id).all()
    created = []
    taken = {l.username for l in Lecturer.query.all()}
    pool = iter(p for p in LECTURER_POOL if p[1] not in taken)
    while len(lecturers) < target:
        try:
            full_name, username = next(pool)
        except StopIteration:
            raise SystemExit(f"  Not enough made-up names: add more to LECTURER_POOL (need {target}).")
        lecturer = Lecturer(username=username, full_name=full_name,
                            email=f"{username}@{DEMO_EMAIL_DOMAIN}", is_active=True)
        lecturer.set_password(password)
        if not dry_run:
            db.session.add(lecturer)
            db.session.flush()
        lecturers.append(lecturer)
        created.append(lecturer)
    return lecturers[:target], created


def main():
    parser = argparse.ArgumentParser(description="Spread demo students across lecturers' sections.")
    parser.add_argument("--lecturers", type=int, default=15, help="number of lecturers (default 15)")
    parser.add_argument("--per-lecturer", type=int, default=4, help="sections per lecturer (default 4)")
    parser.add_argument("--password", help="password for newly created lecturers (default: generated)")
    parser.add_argument("--seed", type=int, default=42, help="change to get a different split")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without saving")
    args = parser.parse_args()
    if args.password is not None and len(args.password) < 8:
        parser.error("--password must be at least 8 characters")
    password = args.password or secrets.token_urlsafe(9)

    with app.app_context():
        demo_sections = [s for s in ClassSection.query.all() if is_demo_section(s)]
        demo_ids = [s.id for s in demo_sections]
        rows = (db.session.query(Student, Prediction.risk_level)
                .outerjoin(Prediction, Prediction.student_id == Student.student_id)
                .filter(Student.section_id.in_(demo_ids or [None]))
                .order_by(Student.student_id).all())
        if not rows:
            print("  Nothing to do: no students in Default Section or earlier demo sections.")
            return

        lecturers, created = ensure_lecturers(args.lecturers, password, args.dry_run)
        total_sections = len(lecturers) * args.per_lecturer
        width = len(str(total_sections))
        names = [f"Section {n:0{width}d}" for n in range(1, total_sections + 1)]

        # Group by risk level, shuffle, then deal round-robin so every section gets a similar mix
        rng = random.Random(args.seed)
        buckets = {level: [] for level in RISK_LEVELS + (None,)}
        for student, level in rows:
            buckets[level if level in RISK_LEVELS else None].append(student)
        dealt = [[] for _ in names]
        i = 0
        for level in RISK_LEVELS + (None,):
            group = buckets[level]
            rng.shuffle(group)
            for student in group:
                dealt[i % len(names)].append(student)
                i += 1

        print(f"  {len(rows)} students → {len(names)} sections, {len(lecturers)} lecturers × "
              f"{args.per_lecturer} sections ({min(map(len, dealt))}–{max(map(len, dealt))} students each)\n")
        plan = []
        for idx, lecturer in enumerate(lecturers):
            mine = list(range(idx * args.per_lecturer, (idx + 1) * args.per_lecturer))
            students = [s for j in mine for s in dealt[j]]
            high = sum(1 for s in students if s.prediction and s.prediction.risk_level == "High Risk")
            new = " (new)" if lecturer in created else ""
            print(f"  {lecturer.full_name + new:<44} {names[mine[0]]} – {names[mine[-1]]}   "
                  f"{len(students):>4} students, {high:>3} high risk")
            plan += [(names[j], lecturer, dealt[j]) for j in mine]

        if args.dry_run:
            print(f"\n  Dry run: nothing saved ({len(created)} lecturer(s) would be created).")
            return

        template = demo_sections[0]
        keep = set()
        for name, lecturer, students in plan:
            section = ClassSection.query.filter_by(lecturer_id=lecturer.id, section_name=name).first()
            if not section:
                section = ClassSection(
                    lecturer_id=lecturer.id, section_name=name,
                    current_semester=template.current_semester, sem_depth=template.sem_depth,
                    # same global Sem 3 model the Default Section used
                    model_path=template.model_path, features_path=template.features_path,
                )
                db.session.add(section)
                db.session.flush()
            keep.add(section.id)
            ids = [s.student_id for s in students]
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                Student.query.filter(Student.student_id.in_(chunk)).update(
                    {Student.section_id: section.id}, synchronize_session=False)
                RiskNotification.query.filter(RiskNotification.student_id.in_(chunk)).update(
                    {RiskNotification.section_id: section.id}, synchronize_session=False)

        removed = 0
        for section in demo_sections:
            if section.id in keep:
                continue
            if Student.query.filter_by(section_id=section.id).count():
                db.session.rollback()
                raise SystemExit(f"  Aborted: {section.section_name} would still have students.")
            db.session.delete(section)
            removed += 1
        db.session.commit()

        print(f"\n  Done. {len(keep)} sections ready, {removed} old section(s) removed.")
        if created:
            print(f"  Created {len(created)} lecturer(s): " + ", ".join(l.username for l in created))
            print(f"  Their password: {password}" + ("" if args.password else "   (generated; save it now)"))


if __name__ == "__main__":
    main()
