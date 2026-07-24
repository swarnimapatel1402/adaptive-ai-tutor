"""
analysis.py — Student Analysis with Pandas
============================================
Run this file to see all students compared
side by side in clean tables.

Usage:
    py analysis.py
"""

import json
import os
import pandas as pd


SAVE_FOLDER = "student_data"

TOPICS = [
    "arithmetic",
    "fractions",
    "algebra_basics",
    "geometry",
    "statistics",
    "linear_equations",
    "quadratics",
    "trigonometry",
    "probability",
    "calculus_intro",
]


def load_all_students() -> list:
    """Load all student JSON files from student_data folder."""
    if not os.path.exists(SAVE_FOLDER):
        print("❌ No student_data folder found. Run main.py first!")
        return []

    files = [f for f in os.listdir(SAVE_FOLDER) if f.endswith(".json")]
    if not files:
        print("❌ No students saved yet. Run main.py first!")
        return []

    students = []
    for file in files:
        with open(os.path.join(SAVE_FOLDER, file), "r") as f:
            students.append(json.load(f))
    return students


def show_summary_table(students: list):
    """
    Table 1 — Summary of all students.
    Shows name, personality, sessions, overall score, mastered topics.
    """
    rows = []
    for s in students:
        knowledge = s["knowledge"]
        overall = round(sum(knowledge.values()) / len(knowledge) * 100, 1)
        mastered = sum(1 for v in knowledge.values() if v >= 0.9)
        in_progress = sum(1 for v in knowledge.values() if 0.1 <= v < 0.9)

        rows.append({
            "Name":        s["name"],
            "Personality": s["personality"].capitalize(),
            "Sessions":    s["sessions"],
            "Overall %":   overall,
            "Mastered":    f"{mastered}/10",
            "In Progress": in_progress,
            "Last Saved":  s.get("last_saved", "unknown"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Overall %", ascending=False).reset_index(drop=True)
    df.index += 1  # start ranking from 1

    print("\n" + "=" * 65)
    print("  📊 Student Summary")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65)


def show_knowledge_table(students: list):
    """
    Table 2 — Knowledge scores per topic for every student.
    Each cell shows the % score for that topic.
    """
    rows = []
    for s in students:
        row = {"Student": s["name"]}
        for topic in TOPICS:
            score = round(s["knowledge"].get(topic, 0) * 100)
            # Add emoji based on score
            if score >= 90:
                row[topic] = f"✅ {score}%"
            elif score >= 10:
                row[topic] = f"🔄 {score}%"
            else:
                row[topic] = f"⬜ {score}%"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Student")

    print("\n" + "=" * 65)
    print("  📚 Knowledge Breakdown by Topic")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65)


def show_strongest_weakest(students: list):
    """
    Table 3 — Each student's strongest and weakest topic.
    """
    rows = []
    for s in students:
        knowledge = s["knowledge"]
        strongest = max(knowledge, key=knowledge.get)
        weakest = min(knowledge, key=knowledge.get)

        rows.append({
            "Student":          s["name"],
            "Strongest Topic":  strongest,
            "Score":            f"{round(knowledge[strongest]*100)}%",
            "Weakest Topic":    weakest,
            "Score ":           f"{round(knowledge[weakest]*100)}%",
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 65)
    print("  💪 Strongest & Weakest Topics per Student")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)


def show_leaderboard(students: list):
    """
    Table 4 — Leaderboard ranked by overall score.
    """
    rows = []
    for s in students:
        knowledge = s["knowledge"]
        overall = round(sum(knowledge.values()) / len(knowledge) * 100, 1)
        mastered = sum(1 for v in knowledge.values() if v >= 0.9)
        rows.append({
            "Student":     s["name"],
            "Overall %":   overall,
            "Mastered":    mastered,
            "Sessions":    s["sessions"],
            "Personality": s["personality"].capitalize(),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Overall %", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    print("\n" + "=" * 65)
    print("  🏆 Leaderboard")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65)


def export_to_csv(students: list):
    """Export all student data to a CSV file."""
    rows = []
    for s in students:
        row = {
            "name":        s["name"],
            "personality": s["personality"],
            "sessions":    s["sessions"],
            "overall_%":   round(
                sum(s["knowledge"].values()) / len(s["knowledge"]) * 100, 1
            ),
        }
        for topic in TOPICS:
            row[topic] = round(s["knowledge"].get(topic, 0) * 100, 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("student_data/all_students.csv", index=False)
    print("\n💾 Data exported → student_data/all_students.csv")


def main():
    print("\n" + "=" * 65)
    print("       📊 Adaptive AI Tutor — Student Analysis")
    print("=" * 65)

    students = load_all_students()
    if not students:
        return

    print(f"\n✅ Loaded {len(students)} student(s) successfully!")

    # Show all 4 tables
    show_summary_table(students)
    show_knowledge_table(students)
    show_strongest_weakest(students)
    show_leaderboard(students)

    # Export to CSV
    export_to_csv(students)


if __name__ == "__main__":
    main()
