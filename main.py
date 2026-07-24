from tutor import AdaptiveAITutor
import os


def main():
    print("\n" + "=" * 40)
    print("       🎓 Adaptive AI Tutor")
    print("=" * 40)

    # ── Step 1: Ask for name ───────────────
    name = input("\nEnter your name: ").strip()

    if not name:
        print("❌ Name cannot be empty!")
        return

    # ── Step 2: New or returning? ──────────
    save_file = f"student_data/{name.lower().replace(' ', '_')}.json"
    is_returning = os.path.exists(save_file)

    if is_returning:
        tutor = AdaptiveAITutor(student_name=name)

    else:
        print(f"\n👋 Welcome {name}! You are a new student.")
        print("\nWhat kind of learner are you?")
        print("  1. Curious  — learns fast, remembers well")
        print("  2. Lazy     — learns slowly, forgets quickly")
        print("  3. Anxious  — struggles without proper order")

        choice = input("\nEnter choice (1/2/3): ").strip()

        personality_map = {"1": "curious", "2": "lazy", "3": "anxious"}
        personality = personality_map.get(choice, "curious")
        tutor = AdaptiveAITutor(student_name=name, personality=personality)

    # ── Step 3: What to do today? ─────────
    while True:
        print("\nWhat do you want to do?")
        print("  1. Auto study (tutor picks best topic)")
        print("  2. Study a specific topic")
        print("  3. See my progress report")
        print("  4. See all students")
        print("  5. Exit")

        action = input("\nEnter choice (1/2/3/4/5): ").strip()

        if action == "1":
            sessions = input("How many sessions? (default 5): ").strip()
            sessions = int(sessions) if sessions.isdigit() else 5
            tutor.auto_study(sessions=sessions)
            tutor.save()

        elif action == "2":
            print("\nAvailable topics:")
            topics = [
                "arithmetic", "fractions", "algebra_basics",
                "geometry", "statistics", "linear_equations",
                "quadratics", "trigonometry", "probability",
                "calculus_intro"
            ]
            for i, topic in enumerate(topics, 1):
                score = tutor.student.knowledge[topic]
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                print(f"  {i:2}. {topic:<20} [{bar}] {round(score*100)}%")

            topic_choice = input("\nEnter topic number: ").strip()
            if topic_choice.isdigit():
                index = int(topic_choice) - 1
                if 0 <= index < len(topics):
                    tutor.teach(topics[index])
                    tutor.save()
                else:
                    print("❌ Invalid number")
            else:
                print("❌ Please enter a number")

        elif action == "3":
            tutor.progress_report()

        elif action == "4":
            AdaptiveAITutor.list_students()

        elif action == "5":
            print(f"\n👋 Goodbye {tutor.student.name}! See you next time!")
            print(f"💾 Your progress is saved.\n")
            break

        else:
            print("❌ Invalid choice — please enter 1, 2, 3, 4 or 5")


if __name__ == "__main__":
    main()
