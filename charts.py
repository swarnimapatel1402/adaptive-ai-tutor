"""
charts.py — Visual Progress Charts with Matplotlib
====================================================
Run this file to see beautiful charts for any student.

Usage:
    py charts.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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

# Short names for chart labels
SHORT_NAMES = [
    "Arith", "Frac", "Algebra",
    "Geo", "Stats", "Lin.Eq",
    "Quad", "Trig", "Prob", "Calc"
]

# Colors based on score
def get_color(score):
    if score >= 0.9:
        return "#2ecc71"   # green  = mastered
    elif score >= 0.5:
        return "#3498db"   # blue   = learning
    elif score >= 0.1:
        return "#f39c12"   # orange = beginner
    else:
        return "#e0e0e0"   # grey   = not started


def load_student(name: str) -> dict:
    """Load a student's data from JSON."""
    path = os.path.join(
        SAVE_FOLDER,
        f"{name.lower().replace(' ', '_')}.json"
    )
    if not os.path.exists(path):
        print(f"❌ No data found for '{name}'. Run main.py first!")
        return None
    with open(path) as f:
        return json.load(f)


def load_all_students() -> list:
    """Load all saved students."""
    if not os.path.exists(SAVE_FOLDER):
        print("❌ No student_data folder found. Run main.py first!")
        return []
    files = [f for f in os.listdir(SAVE_FOLDER) if f.endswith(".json")]
    students = []
    for file in files:
        with open(os.path.join(SAVE_FOLDER, file)) as f:
            students.append(json.load(f))
    return students


# ── Chart 1: Student Knowledge Bar Chart ──────────────────────────────────────

def chart_student_progress(name: str):
    """
    Bar chart showing knowledge score per topic for one student.
    Bars are colour coded by mastery level.
    """
    data = load_student(name)
    if not data:
        return

    scores = [data["knowledge"].get(t, 0) for t in TOPICS]
    colors = [get_color(s) for s in scores]
    overall = round(sum(scores) / len(scores) * 100, 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(SHORT_NAMES, [s * 100 for s in scores],
                  color=colors, edgecolor="white", linewidth=1.5)

    # Add score labels on top of each bar
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{round(score*100)}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    # Add 50% threshold line
    ax.axhline(y=50, color="#e74c3c", linestyle="--",
               linewidth=1.5, alpha=0.7, label="Prerequisite threshold (50%)")
    ax.axhline(y=90, color="#2ecc71", linestyle="--",
               linewidth=1.5, alpha=0.7, label="Mastery threshold (90%)")

    # Legend
    legend_items = [
        mpatches.Patch(color="#2ecc71", label="Mastered (90%+)"),
        mpatches.Patch(color="#3498db", label="Learning (50-89%)"),
        mpatches.Patch(color="#f39c12", label="Beginner (10-49%)"),
        mpatches.Patch(color="#e0e0e0", label="Not started"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Knowledge Score (%)", fontsize=12)
    ax.set_title(
        f"📚 {data['name']} — Knowledge Progress\n"
        f"Personality: {data['personality'].capitalize()} | "
        f"Sessions: {data['sessions']} | "
        f"Overall: {overall}%",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()

    filename = f"student_data/{name.lower()}_progress.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved → {filename}")
    plt.show()


# ── Chart 2: All Students Comparison ──────────────────────────────────────────

def chart_all_students_comparison():
    """
    Grouped bar chart comparing all students across all topics.
    """
    students = load_all_students()
    if not students:
        return

    x = np.arange(len(SHORT_NAMES))
    width = 0.8 / len(students)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12",
              "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, student in enumerate(students):
        scores = [student["knowledge"].get(t, 0) * 100 for t in TOPICS]
        offset = (i - len(students) / 2 + 0.5) * width
        ax.bar(x + offset, scores, width,
               label=student["name"],
               color=colors[i % len(colors)],
               alpha=0.85, edgecolor="white")

    ax.axhline(y=50, color="black", linestyle="--",
               linewidth=1, alpha=0.4, label="50% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Knowledge Score (%)", fontsize=12)
    ax.set_title(
        "📊 All Students — Topic Comparison",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.legend(fontsize=10)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()

    filename = "student_data/all_students_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved → {filename}")
    plt.show()


# ── Chart 3: Leaderboard Bar Chart ────────────────────────────────────────────

def chart_leaderboard():
    """
    Horizontal bar chart showing overall score per student — ranked.
    """
    students = load_all_students()
    if not students:
        return

    # Calculate overall score
    data = []
    for s in students:
        overall = sum(s["knowledge"].values()) / len(s["knowledge"]) * 100
        data.append((s["name"], round(overall, 1), s["personality"]))

    # Sort by score
    data.sort(key=lambda x: x[1], reverse=True)
    names = [d[0] for d in data]
    scores = [d[1] for d in data]
    personalities = [d[2] for d in data]

    personality_colors = {
        "curious": "#3498db",
        "lazy":    "#e74c3c",
        "anxious": "#f39c12",
    }
    colors = [personality_colors.get(p, "#95a5a6") for p in personalities]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 1.2)))
    bars = ax.barh(names, scores, color=colors,
                   edgecolor="white", linewidth=1.5, height=0.5)

    # Score labels
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score}%",
            va="center", fontsize=11, fontweight="bold"
        )

    # Legend
    legend_items = [
        mpatches.Patch(color="#3498db", label="Curious"),
        mpatches.Patch(color="#e74c3c", label="Lazy"),
        mpatches.Patch(color="#f39c12", label="Anxious"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=10)

    ax.set_xlim(0, 115)
    ax.set_xlabel("Overall Score (%)", fontsize=12)
    ax.set_title(
        "🏆 Student Leaderboard",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()

    filename = "student_data/leaderboard.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved → {filename}")
    plt.show()


# ── Chart 4: Radar Chart (Spider Chart) ───────────────────────────────────────

def chart_radar(name: str):
    """
    Radar/spider chart showing a student's knowledge shape.
    """
    data = load_student(name)
    if not data:
        return

    scores = [data["knowledge"].get(t, 0) for t in TOPICS]
    scores_pct = [s * 100 for s in scores]

    # Close the loop for radar chart
    scores_pct += scores_pct[:1]
    angles = np.linspace(0, 2 * np.pi, len(TOPICS),
                         endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))

    ax.fill(angles, scores_pct, color="#3498db", alpha=0.25)
    ax.plot(angles, scores_pct, color="#3498db",
            linewidth=2, linestyle="solid")

    # Add threshold rings
    ax.plot(angles, [50] * len(angles), color="#e74c3c",
            linewidth=1, linestyle="--", alpha=0.5)
    ax.plot(angles, [90] * len(angles), color="#2ecc71",
            linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(SHORT_NAMES, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)

    overall = round(sum(scores) / len(scores) * 100, 1)
    ax.set_title(
        f"🕸️ {data['name']} — Knowledge Radar\n"
        f"Overall: {overall}% | Sessions: {data['sessions']}",
        fontsize=13, fontweight="bold", pad=20
    )

    filename = f"student_data/{name.lower()}_radar.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved → {filename}")
    plt.show()


# ── Main Menu ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 45)
    print("   📊 Adaptive AI Tutor — Charts")
    print("=" * 45)

    # Show available students
    students = load_all_students()
    if not students:
        return

    print(f"\n✅ Found {len(students)} student(s):")
    for s in students:
        overall = round(
            sum(s["knowledge"].values()) / len(s["knowledge"]) * 100, 1
        )
        print(f"   👤 {s['name']} — {overall}% overall")

    print("\nWhat chart do you want?")
    print("  1. My progress bar chart     (one student)")
    print("  2. All students comparison   (everyone together)")
    print("  3. Leaderboard chart         (ranked by score)")
    print("  4. My radar/spider chart     (one student)")
    print("  5. All charts at once")

    choice = input("\nEnter choice (1/2/3/4/5): ").strip()

    if choice in ["1", "4"]:
        name = input("Enter student name: ").strip()
        if choice == "1":
            chart_student_progress(name)
        else:
            chart_radar(name)

    elif choice == "2":
        chart_all_students_comparison()

    elif choice == "3":
        chart_leaderboard()

    elif choice == "5":
        name = input("Enter your name for personal charts: ").strip()
        print("\n🎨 Generating all charts...\n")
        chart_student_progress(name)
        chart_all_students_comparison()
        chart_leaderboard()
        chart_radar(name)
        print("\n✅ All charts saved to student_data folder!")

    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()