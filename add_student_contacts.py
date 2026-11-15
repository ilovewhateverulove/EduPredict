"""
EduPredict — add_student_contacts.py
════════════════════════════════════════════════════════════════════
Adds made-up FullName and Email columns to the demo dataset so the
web app can show student names and send risk-alert emails.

Names follow each student's Gender and Ethnicity. The same StudentID
always gets the same name, so re-running is safe.

Usage:
  python add_student_contacts.py
      → emails like aisyah.1001@student.edupredict.test (never delivered)

  python add_student_contacts.py --email yourname@gmail.com
      → emails like yourname+1001@gmail.com (all arrive in your inbox,
        handy for demoing risk alerts)

Updates:
  data/Student_Performance_Enhanced.xlsx
  outputs/predictions.csv

FullName and Email are never used as model features.
════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import random
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = os.path.join(BASE_DIR, "data", "Student_Performance_Enhanced.xlsx")
PRED_CSV  = os.path.join(BASE_DIR, "outputs", "predictions.csv")

NAMES = {
    "Malay": {
        "Male":   ["Muhammad Hafiz", "Ahmad Faiz", "Muhammad Amir", "Mohd Firdaus", "Muhammad Haziq",
                   "Adam Danial", "Muhammad Irfan", "Aiman Hakimi", "Syafiq Aizat", "Muhammad Luqman",
                   "Iqbal Hakim", "Harith Iskandar", "Zulfikri", "Nazrul Hisyam", "Akmal Hazim"],
        "Female": ["Nur Aisyah", "Siti Nurhaliza", "Nurul Izzah", "Nur Syafiqah", "Aina Sofea",
                   "Nur Farhana", "Siti Aminah", "Nurin Batrisyia", "Alya Maisarah", "Nur Hidayah",
                   "Puteri Balqis", "Hani Sabrina", "Nur Amirah", "Iman Natasha", "Fatin Nabila"],
        "fathers": ["Ahmad", "Abdullah", "Ismail", "Rahman", "Hassan", "Ibrahim", "Yusof", "Osman",
                    "Razak", "Kamarudin", "Zainal", "Hamid", "Rosli", "Azman", "Salleh"],
    },
    "Chinese": {
        "Male":   ["Wei Ming", "Jia Hao", "Zhi Wei", "Kok Leong", "Jun Kai", "Wei Jie", "Chee Keong",
                   "Yong Sheng", "Kah Seng", "Jian Hong", "Boon Hock", "Zi Xuan"],
        "Female": ["Mei Ling", "Hui Min", "Xin Yi", "Siew Lan", "Jia Yi", "Pei Shan", "Li Ying",
                   "Shu Fen", "Wen Qi", "Kai Xin", "Yee Ting", "Zi Ning"],
        "surnames": ["Tan", "Lim", "Lee", "Wong", "Ng", "Chan", "Ong", "Goh", "Teoh", "Chong",
                     "Yap", "Low", "Koh", "Chin", "Loh"],
    },
    "Indian": {
        "Male":   ["Arjun", "Ravi", "Suresh", "Karthik", "Vikneswaran", "Prakash", "Dinesh",
                   "Harish", "Naveen", "Thinesh", "Sanjay", "Kumaran"],
        "Female": ["Kavitha", "Priya", "Shalini", "Deepa", "Thivya", "Nisha", "Anusha",
                   "Lavanya", "Revathi", "Keerthana", "Sharmila", "Yuvasri"],
        "fathers": ["Rajan", "Subramaniam", "Muthu", "Krishnan", "Ganesan", "Raman", "Selvam",
                    "Murugan", "Gopal", "Balakrishnan", "Ramasamy", "Nadarajan"],
    },
    "Bumiputera Sabah/Sarawak": {
        "Male":   ["Jerry", "Mojuntin", "Benedict", "Alvin", "Jimbun", "Rayner", "Gilbert",
                   "Juanis", "Kelvin", "Mathius", "Ronny", "Edwin"],
        "Female": ["Jacqueline", "Ruminah", "Grace", "Velerie", "Linda", "Juliana", "Melissa",
                   "Agnes", "Christina", "Rosalind", "Elvina", "Joanna"],
        "fathers": ["Jimbun", "Gindol", "Tingkoi", "Ngelai", "Kitingan", "Majimbun", "Sulang",
                    "Entalai", "Lajim", "Masidi", "Unggang", "Dumpangol"],
    },
}


def make_name(rng: random.Random, gender: str, ethnicity: str):
    """Return (full_name, short_name_for_email)."""
    gender = gender if gender in ("Male", "Female") else rng.choice(["Male", "Female"])
    group = NAMES.get(ethnicity, NAMES["Malay"])
    given = rng.choice(group[gender])
    if ethnicity == "Chinese":
        surname = rng.choice(group["surnames"])
        return f"{surname} {given}", f"{given.split()[0]}.{surname}"
    father = rng.choice(group["fathers"])
    if ethnicity == "Indian":
        link = "a/l" if gender == "Male" else "a/p"
    elif ethnicity == "Bumiputera Sabah/Sarawak":
        link = "anak"
    else:
        link = "bin" if gender == "Male" else "binti"
    return f"{given} {link} {father}", given.split()[-1]


def build_contacts(df: pd.DataFrame, email_base: str | None, seed: int) -> pd.DataFrame:
    names, emails = [], []
    for _, row in df.iterrows():
        sid = str(row["StudentID"]).strip()
        rng = random.Random(f"{seed}-{sid}")
        full_name, short = make_name(rng, str(row.get("Gender", "")), str(row.get("Ethnicity", "")))
        if email_base:
            local, domain = email_base.split("@", 1)
            email = f"{local}+{sid}@{domain}"
        else:
            slug = "".join(ch for ch in short.lower() if ch.isalnum() or ch == ".")
            email = f"{slug}.{sid}@student.edupredict.test"
        names.append(full_name)
        emails.append(email.lower())
    return pd.DataFrame({"StudentID": df["StudentID"], "FullName": names, "Email": emails})


def insert_contacts(df: pd.DataFrame, contacts: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in ("FullName", "Email") if c in df.columns])
    lookup = contacts.set_index(contacts["StudentID"].astype(str).str.strip())
    key = df["StudentID"].astype(str).str.strip()
    pos = df.columns.get_loc("StudentID") + 1
    added = pd.DataFrame({"FullName": key.map(lookup["FullName"]).values,
                          "Email": key.map(lookup["Email"]).values}, index=df.index)
    return pd.concat([df.iloc[:, :pos], added, df.iloc[:, pos:]], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Add made-up student names and emails to the dataset.")
    parser.add_argument("--email", help="your own address; students become yourname+ID@domain")
    parser.add_argument("--seed", type=int, default=42, help="change to get a different set of names")
    args = parser.parse_args()
    if args.email and "@" not in args.email:
        parser.error("--email must look like yourname@gmail.com")

    sheets = pd.read_excel(DATA_XLSX, sheet_name=None)
    first = next(iter(sheets))
    contacts = build_contacts(sheets[first], args.email, args.seed)
    sheets[first] = insert_contacts(sheets[first], contacts)
    with pd.ExcelWriter(DATA_XLSX) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    print(f"  [OK] {len(contacts)} students updated in {os.path.relpath(DATA_XLSX, BASE_DIR)}")

    if os.path.isfile(PRED_CSV):
        preds = insert_contacts(pd.read_csv(PRED_CSV), contacts)
        preds.to_csv(PRED_CSV, index=False)
        print(f"  [OK] {len(preds)} rows updated in {os.path.relpath(PRED_CSV, BASE_DIR)}")

    print("\n  Sample:")
    print(contacts.head(5).to_string(index=False))
    print("\n  Restart the app: students already in the database get these names automatically.")


if __name__ == "__main__":
    main()
