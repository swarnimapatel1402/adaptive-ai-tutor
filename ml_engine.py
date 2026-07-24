"""
ml_engine.py — Adaptive AI Tutor
Three ML models:
  1. Performance Predictor  → Linear Regression  (predicts next score)
  2. Difficulty Recommender → Decision Tree      (recommends difficulty level)
  3. Knowledge Gap Detector → KMeans Clustering  (finds weak topics)
"""

import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


# ─────────────────────────────────────────────
#  DATA FILE PATH  (matches your json setup)
# ─────────────────────────────────────────────
DATA_FILE = "student_data.json"


def load_student_data():
    """Load student session data from JSON file."""
    if not os.path.exists(DATA_FILE):
        print(f"[ML] No data file found at '{DATA_FILE}'. Using sample data.")
        return _sample_data()
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_student_data(data):
    """Save updated student data back to JSON."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


def _sample_data():
    """
    Fallback sample data — used when no real student sessions exist yet.
    Each entry = one quiz/session result.
    Fields:
      student_id, topic, difficulty (0=easy,1=medium,2=hard),
      score (0-100), time_taken (seconds), session_number
    """
    return [
        {"student_id": "s1", "topic": "math",    "difficulty": 0, "score": 80, "time_taken": 45,  "session_number": 1},
        {"student_id": "s1", "topic": "math",    "difficulty": 1, "score": 65, "time_taken": 70,  "session_number": 2},
        {"student_id": "s1", "topic": "science",  "difficulty": 0, "score": 90, "time_taken": 30,  "session_number": 3},
        {"student_id": "s1", "topic": "history",  "difficulty": 1, "score": 55, "time_taken": 90,  "session_number": 4},
        {"student_id": "s1", "topic": "math",    "difficulty": 2, "score": 50, "time_taken": 120, "session_number": 5},
        {"student_id": "s1", "topic": "science",  "difficulty": 1, "score": 75, "time_taken": 60,  "session_number": 6},
        {"student_id": "s1", "topic": "english",  "difficulty": 0, "score": 85, "time_taken": 40,  "session_number": 7},
        {"student_id": "s1", "topic": "history",  "difficulty": 0, "score": 60, "time_taken": 80,  "session_number": 8},
        {"student_id": "s1", "topic": "english",  "difficulty": 1, "score": 70, "time_taken": 55,  "session_number": 9},
        {"student_id": "s1", "topic": "math",    "difficulty": 1, "score": 72, "time_taken": 65,  "session_number": 10},
        {"student_id": "s2", "topic": "math",    "difficulty": 0, "score": 60, "time_taken": 80,  "session_number": 1},
        {"student_id": "s2", "topic": "science",  "difficulty": 0, "score": 55, "time_taken": 95,  "session_number": 2},
        {"student_id": "s2", "topic": "history",  "difficulty": 1, "score": 45, "time_taken": 110, "session_number": 3},
        {"student_id": "s2", "topic": "english",  "difficulty": 0, "score": 78, "time_taken": 50,  "session_number": 4},
        {"student_id": "s2", "topic": "math",    "difficulty": 1, "score": 50, "time_taken": 100, "session_number": 5},
    ]


# ═══════════════════════════════════════════════════════════════
#  MODEL 1 — PERFORMANCE PREDICTOR  (Linear Regression)
#  Predicts what score a student will get in their next session.
# ═══════════════════════════════════════════════════════════════

class PerformancePredictor:
    """
    Predicts a student's next quiz score using Linear Regression.
    Features: session_number, difficulty, time_taken
    Target:   score (0-100)
    """

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.trained = False

    def _prepare_features(self, data):
        X, y = [], []
        for record in data:
            X.append([
                record["session_number"],
                record["difficulty"],
                record["time_taken"]
            ])
            y.append(record["score"])
        return np.array(X), np.array(y)

    def train(self, data):
        if len(data) < 3:
            print("[PerformancePredictor] Not enough data (need >= 3 records).")
            return

        X, y = self._prepare_features(data)
        X_scaled = self.scaler.fit_transform(X)

        if len(data) >= 6:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            print(f"[PerformancePredictor] Trained. RMSE: {rmse:.2f}")
        else:
            self.model.fit(X_scaled, y)
            print("[PerformancePredictor] Trained on all data (small dataset).")

        self.trained = True

    def predict_next_score(self, session_number, difficulty, time_taken):
        """Returns predicted score (float, clamped 0-100)."""
        if not self.trained:
            print("[PerformancePredictor] Not trained yet. Call train() first.")
            return None
        features = np.array([[session_number, difficulty, time_taken]])
        features_scaled = self.scaler.transform(features)
        predicted = self.model.predict(features_scaled)[0]
        return max(0, min(100, round(predicted, 2)))


# ═══════════════════════════════════════════════════════════════
#  MODEL 2 — DIFFICULTY RECOMMENDER  (Decision Tree)
#  Recommends easy / medium / hard for the student's next session.
# ═══════════════════════════════════════════════════════════════

class DifficultyRecommender:
    """
    Recommends difficulty using a Decision Tree.
    Features: last_score, avg_score, time_taken, current_difficulty
    Target:   next difficulty (0=easy, 1=medium, 2=hard)
    """

    LABELS = {0: "Easy", 1: "Medium", 2: "Hard"}

    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.trained = False

    def _build_training_data(self, data):
        X, y = [], []
        sessions = sorted(data, key=lambda r: r["session_number"])
        for i in range(len(sessions) - 1):
            curr = sessions[i]
            nxt  = sessions[i + 1]
            avg_score = np.mean([r["score"] for r in sessions[:i+1]])
            X.append([curr["score"], avg_score, curr["time_taken"], curr["difficulty"]])

            # Label: what difficulty SHOULD come next?
            if nxt["score"] >= 75:
                label = min(curr["difficulty"] + 1, 2)
            elif nxt["score"] < 50:
                label = max(curr["difficulty"] - 1, 0)
            else:
                label = curr["difficulty"]
            y.append(label)
        return np.array(X), np.array(y)

    def train(self, data):
        sessions = sorted(data, key=lambda r: r["session_number"])
        if len(sessions) < 3:
            print("[DifficultyRecommender] Need >= 3 sessions to train.")
            return

        X, y = self._build_training_data(sessions)
        if len(X) == 0:
            return

        self.model.fit(X, y)
        self.trained = True
        acc = accuracy_score(y, self.model.predict(X))
        print(f"[DifficultyRecommender] Trained. Accuracy: {acc*100:.1f}%")

    def recommend(self, last_score, avg_score, time_taken, current_difficulty):
        """Returns (int, str) e.g. (2, 'Hard')."""
        if not self.trained:
            # Simple rule-based fallback
            if last_score >= 80:
                rec = min(current_difficulty + 1, 2)
            elif last_score < 50:
                rec = max(current_difficulty - 1, 0)
            else:
                rec = current_difficulty
            print("[DifficultyRecommender] Using rule-based fallback.")
            return rec, self.LABELS[rec]

        features = np.array([[last_score, avg_score, time_taken, current_difficulty]])
        rec = int(self.model.predict(features)[0])
        return rec, self.LABELS[rec]


# ═══════════════════════════════════════════════════════════════
#  MODEL 3 — KNOWLEDGE GAP DETECTOR  (KMeans Clustering)
#  Groups topics into Strong / Average / Weak clusters.
# ═══════════════════════════════════════════════════════════════

class KnowledgeGapDetector:
    """
    Clusters topics based on avg score and avg time per topic.
    Cluster with highest avg_score → Strong
    Cluster with lowest  avg_score → Weak
    """

    N_CLUSTERS = 3

    def __init__(self):
        self.model = KMeans(n_clusters=self.N_CLUSTERS, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.topic_stats = {}
        self.cluster_labels = {}
        self.trained = False

    def _compute_topic_stats(self, data):
        topic_data = {}
        for record in data:
            t = record["topic"]
            if t not in topic_data:
                topic_data[t] = {"scores": [], "times": []}
            topic_data[t]["scores"].append(record["score"])
            topic_data[t]["times"].append(record["time_taken"])

        stats = {}
        for topic, vals in topic_data.items():
            stats[topic] = {
                "avg_score": np.mean(vals["scores"]),
                "avg_time":  np.mean(vals["times"]),
                "sessions":  len(vals["scores"])
            }
        return stats

    def _label_clusters(self, stats):
        cluster_scores = {}
        for topic, info in stats.items():
            c = info["cluster"]
            cluster_scores.setdefault(c, []).append(info["avg_score"])

        cluster_means = {c: np.mean(scores) for c, scores in cluster_scores.items()}
        sorted_clusters = sorted(cluster_means, key=cluster_means.get, reverse=True)

        labels = {}
        label_names = ["Strong", "Average", "Weak"]
        for i, cluster_id in enumerate(sorted_clusters):
            labels[cluster_id] = label_names[min(i, 2)]
        return labels

    def train(self, data):
        stats = self._compute_topic_stats(data)
        if len(stats) < self.N_CLUSTERS:
            print(f"[KnowledgeGapDetector] Need >= {self.N_CLUSTERS} topics. Skipping clustering.")
            self.topic_stats = stats
            return

        topics  = list(stats.keys())
        X = np.array([[stats[t]["avg_score"], stats[t]["avg_time"]] for t in topics])
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)
        self.trained = True

        for i, topic in enumerate(topics):
            stats[topic]["cluster"] = int(self.model.labels_[i])

        self.topic_stats   = stats
        self.cluster_labels = self._label_clusters(stats)
        print(f"[KnowledgeGapDetector] Trained. {len(topics)} topics clustered.")

    def get_gaps(self):
        """Returns {'Strong': [...], 'Average': [...], 'Weak': [...]}."""
        if not self.trained or not self.topic_stats:
            return {"Strong": [], "Average": [], "Weak": []}

        result = {"Strong": [], "Average": [], "Weak": []}
        for topic, info in self.topic_stats.items():
            c = info.get("cluster")
            if c is not None:
                label = self.cluster_labels.get(c, "Average")
                result[label].append({
                    "topic":     topic,
                    "avg_score": round(info["avg_score"], 1),
                    "avg_time":  round(info["avg_time"], 1),
                    "sessions":  info["sessions"]
                })
        return result

    def recommend_focus(self):
        """Returns list of weak topic names."""
        return [item["topic"] for item in self.get_gaps().get("Weak", [])]


# ═══════════════════════════════════════════════════════════════
#  ML ENGINE — unified interface for tutor.py and app.py
# ═══════════════════════════════════════════════════════════════

class MLEngine:
    """
    Single entry point for all 3 ML models.

    Typical usage in tutor.py / app.py:

        from ml_engine import MLEngine
        ml = MLEngine()
        ml.load_and_train(student_id="s1")

        score = ml.predict_score(session_number=5, difficulty=1, time_taken=60)
        diff_int, diff_label = ml.recommend_difficulty(72, 68, 65, 1)
        gaps  = ml.get_knowledge_gaps()
        focus = ml.get_focus_topics()
        ml.full_report("s1")

        # After a quiz:
        ml.add_session("s1", "math", difficulty=1, score=74, time_taken=55)
    """

    def __init__(self):
        self.predictor    = PerformancePredictor()
        self.recommender  = DifficultyRecommender()
        self.gap_detector = KnowledgeGapDetector()
        self.data = []

    def load_and_train(self, student_id=None):
        """Load JSON data and train all 3 models."""
        all_data = load_student_data()

        if student_id:
            self.data = [r for r in all_data if r.get("student_id") == student_id]
            if not self.data:
                print(f"[MLEngine] No data for '{student_id}'. Using all data.")
                self.data = all_data
        else:
            self.data = all_data

        print(f"\n[MLEngine] Training on {len(self.data)} session records...\n")
        self.predictor.train(self.data)
        self.recommender.train(self.data)
        self.gap_detector.train(self.data)
        print("\n[MLEngine] All 3 models ready.\n")

    def add_session(self, student_id, topic, difficulty, score, time_taken):
        """
        Save a new session result and retrain models.
        Call this after every quiz in tutor.py.
        """
        all_data = load_student_data()
        student_sessions = [r for r in all_data if r.get("student_id") == student_id]
        new_record = {
            "student_id":     student_id,
            "topic":          topic,
            "difficulty":     difficulty,
            "score":          score,
            "time_taken":     time_taken,
            "session_number": len(student_sessions) + 1
        }
        all_data.append(new_record)
        save_student_data(all_data)
        print(f"[MLEngine] Session saved → {student_id} | {topic} | score={score}")
        self.load_and_train(student_id=student_id)

    def predict_score(self, session_number, difficulty, time_taken):
        """Predict next score. Returns float."""
        return self.predictor.predict_next_score(session_number, difficulty, time_taken)

    def recommend_difficulty(self, last_score, avg_score, time_taken, current_difficulty):
        """Returns (int, str) e.g. (2, 'Hard')."""
        return self.recommender.recommend(last_score, avg_score, time_taken, current_difficulty)

    def get_knowledge_gaps(self):
        """Returns {'Strong': [...], 'Average': [...], 'Weak': [...]}."""
        return self.gap_detector.get_gaps()

    def get_focus_topics(self):
        """Returns list of weak topic names."""
        return self.gap_detector.recommend_focus()

    def full_report(self, student_id):
        """Print complete ML analysis for a student."""
        print(f"\n{'='*52}")
        print(f"  ML REPORT — Student: {student_id}")
        print(f"{'='*52}")

        if not self.data:
            print("  No data available.")
            return

        scores    = [r["score"] for r in self.data]
        times     = [r["time_taken"] for r in self.data]
        last      = sorted(self.data, key=lambda r: r["session_number"])[-1]
        avg_score = round(np.mean(scores), 1)
        diff_map  = {0: "Easy", 1: "Medium", 2: "Hard"}

        predicted = self.predict_score(
            last["session_number"] + 1, last["difficulty"], np.mean(times)
        )
        rec_int, rec_label = self.recommend_difficulty(
            last["score"], avg_score, last["time_taken"], last["difficulty"]
        )
        gaps  = self.get_knowledge_gaps()
        focus = self.get_focus_topics()

        print(f"\n  Sessions completed : {len(self.data)}")
        print(f"  Average score      : {avg_score}/100")
        print(f"  Last score         : {last['score']}/100 ({diff_map[last['difficulty']]})")
        print(f"\n  [Model 1] Predicted next score   : {predicted}/100")
        print(f"  [Model 2] Recommended difficulty : {rec_label}")
        print(f"\n  [Model 3] Topic Analysis:")
        for label in ["Strong", "Average", "Weak"]:
            items = gaps.get(label, [])
            topics_str = ", ".join(f"{i['topic']} ({i['avg_score']})" for i in items)
            print(f"    {label:8s}: {topics_str if topics_str else 'none'}")
        if focus:
            print(f"\n  Focus topics : {', '.join(focus)}")
        print(f"{'='*52}\n")


# ═══════════════════════════════════════════════════════════════
#  QUICK TEST — run:  python ml_engine.py
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ml = MLEngine()
    ml.load_and_train(student_id="s1")
    ml.full_report(student_id="s1")

    print("--- Standalone predictions ---")
    score = ml.predict_score(session_number=11, difficulty=1, time_taken=60)
    print(f"Predicted score: {score}/100")

    diff_int, diff_label = ml.recommend_difficulty(
        last_score=72, avg_score=68, time_taken=65, current_difficulty=1
    )
    print(f"Recommended difficulty: {diff_label}")
    print(f"Weak topics to focus on: {ml.get_focus_topics()}")
