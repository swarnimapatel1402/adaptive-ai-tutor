"""
personality_detector.py — Adaptive AI Tutor
=============================================
Detects student personality automatically using Random Forest.

Personalities:
  - curious  → high scores, fast, consistent, explores many topics
  - lazy     → low scores, slow, inconsistent, few sessions
  - anxious  → drops when difficulty rises, high time variance

Usage:
    from personality_detector import PersonalityDetector
    pd = PersonalityDetector()
    pd.train()
    personality = pd.detect("priya")
"""

import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


# ── Data file (same as ml_engine.py) ──────────────────────────────────────────
DATA_FILE      = "student_data.json"
STUDENT_FOLDER = "student_data"        # folder where priya.json etc. are saved


# ── Personality labels ─────────────────────────────────────────────────────────
PERSONALITIES  = ["curious", "lazy", "anxious"]


def _load_ml_sessions():
    """Load session records from student_data.json (ML engine data)."""
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def _load_tutor_file(student_name: str) -> dict:
    """Load a student's tutor save file (knowledge scores, history etc.)."""
    safe  = student_name.lower().replace(" ", "_")
    path  = os.path.join(STUDENT_FOLDER, f"{safe}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTOR
#  Converts raw session data into numbers the ML model can understand.
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(student_id: str, sessions: list) -> np.ndarray:
    """
    Build a feature vector for one student from their session history.

    Features (6 total):
      1. avg_score       — overall performance level
      2. score_trend     — are they improving (+) or declining (-)
      3. avg_time        — average seconds per session
      4. time_variance   — how consistent their time is
      5. topics_tried    — how many different topics attempted
      6. session_count   — total sessions completed
    """
    if not sessions:
        return np.zeros(6)

    scores  = [s["score"]      for s in sessions]
    times   = [s["time_taken"] for s in sessions]
    topics  = [s["topic"]      for s in sessions]

    avg_score     = np.mean(scores)
    score_trend   = (scores[-1] - scores[0]) if len(scores) > 1 else 0
    avg_time      = np.mean(times)
    time_variance = np.std(times)
    topics_tried  = len(set(topics))
    session_count = len(sessions)

    return np.array([
        avg_score,
        score_trend,
        avg_time,
        time_variance,
        topics_tried,
        session_count
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED LABELLER
#  Since we don't have labelled personality data yet, we generate labels
#  from the student's known personality in their tutor save file.
#  If unknown → infer from behaviour patterns.
# ══════════════════════════════════════════════════════════════════════════════

def infer_label_from_behaviour(features: np.ndarray) -> str:
    """
    Rule-based fallback to label a student when no personality is saved.
    Used to generate training data for the Random Forest.

    Rules:
      curious  → high avg score (>70) AND positive trend AND many topics
      lazy     → low avg score (<55) OR very slow time AND few sessions
      anxious  → high time variance AND declining score trend
    """
    avg_score, score_trend, avg_time, time_variance, topics_tried, session_count = features

    if time_variance > 30 and score_trend < -5:
        return "anxious"
    elif avg_score > 70 and topics_tried >= 3 and score_trend >= 0:
        return "curious"
    elif avg_score < 55 or (avg_time > 90 and session_count < 5):
        return "lazy"
    elif score_trend > 10:
        return "curious"
    else:
        return "anxious"


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TRAINING DATA GENERATOR
#  Creates realistic fake student profiles to train the Random Forest.
#  This solves the cold-start problem — we don't need 100 real students
#  to train the model.
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_per_class: int = 40):
    """
    Generate synthetic (avg_score, trend, avg_time, time_var, topics, sessions)
    for each personality type.
    """
    np.random.seed(42)
    X, y = [], []

    for _ in range(n_per_class):
        # CURIOUS — high scores, improving, fast, consistent, many topics
        X.append([
            np.random.uniform(72, 95),    # avg_score
            np.random.uniform(5, 20),     # score_trend (improving)
            np.random.uniform(25, 55),    # avg_time (fast)
            np.random.uniform(3, 15),     # time_variance (consistent)
            np.random.randint(4, 8),      # topics_tried (explores a lot)
            np.random.randint(8, 20),     # session_count (regular)
        ])
        y.append("curious")

        # LAZY — low scores, flat/declining, slow, many sessions skipped
        X.append([
            np.random.uniform(35, 58),    # avg_score (low)
            np.random.uniform(-10, 3),    # score_trend (flat or declining)
            np.random.uniform(80, 130),   # avg_time (slow)
            np.random.uniform(10, 30),    # time_variance
            np.random.randint(1, 4),      # topics_tried (sticks to few)
            np.random.randint(2, 8),      # session_count (low engagement)
        ])
        y.append("lazy")

        # ANXIOUS — inconsistent time, drops on hard topics, medium scores
        X.append([
            np.random.uniform(50, 72),    # avg_score (medium)
            np.random.uniform(-15, 5),    # score_trend (declining or flat)
            np.random.uniform(55, 95),    # avg_time (medium-slow)
            np.random.uniform(30, 60),    # time_variance (very inconsistent!)
            np.random.randint(2, 6),      # topics_tried
            np.random.randint(4, 12),     # session_count
        ])
        y.append("anxious")

    return np.array(X), np.array(y)


# ══════════════════════════════════════════════════════════════════════════════
#  PERSONALITY DETECTOR — main class
# ══════════════════════════════════════════════════════════════════════════════

class PersonalityDetector:
    """
    Detects student personality using a Random Forest Classifier.

    Typical usage:
        detector = PersonalityDetector()
        detector.train()
        result = detector.detect("priya")
        print(result)
        # {
        #   "personality": "curious",
        #   "confidence":  87.5,
        #   "probabilities": {"curious": 87.5, "lazy": 5.0, "anxious": 7.5},
        #   "explanation": "..."
        # }
    """

    def __init__(self):
        self.model   = RandomForestClassifier(
            n_estimators=100,   # 100 decision trees
            max_depth=5,        # prevent overfitting on small data
            random_state=42
        )
        self.encoder = LabelEncoder()
        self.trained = False

    def train(self):
        """
        Train the Random Forest on:
        1. Synthetic data (always available — solves cold start)
        2. Real student data from student_data.json (if available)
        """
        print("[PersonalityDetector] Generating training data...")

        # Step 1: synthetic data
        X_syn, y_syn = generate_synthetic_data(n_per_class=40)

        # Step 2: real student data (if exists)
        X_real, y_real = self._load_real_training_data()

        # Step 3: combine
        if len(X_real) > 0:
            X = np.vstack([X_syn, X_real])
            y = np.concatenate([y_syn, y_real])
            print(f"[PersonalityDetector] Using {len(X_syn)} synthetic + "
                  f"{len(X_real)} real records.")
        else:
            X, y = X_syn, y_syn
            print(f"[PersonalityDetector] Using {len(X_syn)} synthetic records "
                  f"(no real data yet).")

        # Step 4: encode labels (curious=0, lazy=1, anxious=2)
        y_encoded = self.encoder.fit_transform(y)

        # Step 5: train
        self.model.fit(X, y_encoded)
        self.trained = True

        # Step 6: cross-validation score
        scores = cross_val_score(self.model, X, y_encoded, cv=3)
        print(f"[PersonalityDetector] Trained. CV Accuracy: "
              f"{scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

    def _load_real_training_data(self):
        """
        Load real student session data and pair it with known personality labels
        from their tutor save files.
        """
        all_sessions = _load_ml_sessions()
        if not all_sessions:
            return np.array([]), np.array([])

        # Group sessions by student
        by_student = {}
        for record in all_sessions:
            sid = record.get("student_id", "unknown")
            by_student.setdefault(sid, []).append(record)

        X, y = [], []
        for student_id, sessions in by_student.items():
            if len(sessions) < 3:
                continue  # not enough data to characterise this student

            features = extract_features(student_id, sessions)

            # Try to get known personality from tutor save file
            tutor_data  = _load_tutor_file(student_id)
            personality = tutor_data.get("personality", None)

            if personality not in PERSONALITIES:
                # Infer from behaviour if not in save file
                personality = infer_label_from_behaviour(features)

            X.append(features)
            y.append(personality)

        return np.array(X), np.array(y)

    def detect(self, student_id: str) -> dict:
        """
        Detect personality for a specific student.

        Returns a dict with:
          - personality   : detected label (curious/lazy/anxious)
          - confidence    : how sure the model is (%)
          - probabilities : % for each personality
          - explanation   : human-readable reason
          - features      : raw feature values used

        Falls back to rule-based detection if not enough session data.
        """
        if not self.trained:
            print("[PersonalityDetector] Not trained. Call train() first.")
            return {"personality": "curious", "confidence": 0, "explanation": "Not trained"}

        all_sessions = _load_ml_sessions()
        sessions     = [s for s in all_sessions if s.get("student_id") == student_id]

        features = extract_features(student_id, sessions)

        if len(sessions) < 3:
            # Not enough real data → rule-based fallback
            personality = infer_label_from_behaviour(features)
            return {
                "personality":   personality,
                "confidence":    60.0,
                "probabilities": {p: (60.0 if p == personality else 20.0)
                                  for p in PERSONALITIES},
                "explanation":   self._explain(personality, features),
                "features":      self._feature_dict(features),
                "method":        "rule-based (not enough sessions yet)"
            }

        # ML prediction
        X          = features.reshape(1, -1)
        probs      = self.model.predict_proba(X)[0]
        pred_idx   = np.argmax(probs)
        personality = self.encoder.inverse_transform([pred_idx])[0]
        confidence  = round(probs[pred_idx] * 100, 1)

        prob_dict = {}
        for i, label in enumerate(self.encoder.classes_):
            name = self.encoder.inverse_transform([i])[0]
            prob_dict[name] = round(probs[i] * 100, 1)

        return {
            "personality":   personality,
            "confidence":    confidence,
            "probabilities": prob_dict,
            "explanation":   self._explain(personality, features),
            "features":      self._feature_dict(features),
            "method":        "Random Forest"
        }

    def detect_and_update(self, student_id: str) -> str:
        """
        Detect personality AND update the student's tutor save file.
        Call this after every few sessions to keep personality fresh.
        Returns the detected personality string.
        """
        result = self.detect(student_id)
        personality = result["personality"]

        # Update tutor save file
        safe = student_id.lower().replace(" ", "_")
        path = os.path.join(STUDENT_FOLDER, f"{safe}.json")

        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)

            old = data.get("personality", "unknown")
            data["personality"] = personality

            with open(path, "w") as f:
                json.dump(data, f, indent=4)

            if old != personality:
                print(f"[PersonalityDetector] {student_id}: "
                      f"{old} → {personality} "
                      f"(confidence: {result['confidence']}%)")
            else:
                print(f"[PersonalityDetector] {student_id}: "
                      f"still {personality} "
                      f"(confidence: {result['confidence']}%)")

        return personality

    def full_report(self, student_id: str):
        """Print a detailed personality detection report."""
        result = self.detect(student_id)
        f      = result["features"]

        print(f"\n{'='*52}")
        print(f"  PERSONALITY REPORT — {student_id}")
        print(f"{'='*52}")
        print(f"  Detected   : {result['personality'].upper()}")
        print(f"  Confidence : {result['confidence']}%")
        print(f"  Method     : {result['method']}")
        print(f"\n  Probabilities:")
        for name, pct in result["probabilities"].items():
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {name:<8} [{bar}] {pct}%")
        print(f"\n  Feature Analysis:")
        print(f"    Avg score      : {f['avg_score']:.1f}/100")
        print(f"    Score trend    : {f['score_trend']:+.1f} (+ = improving)")
        print(f"    Avg time       : {f['avg_time']:.0f}s per session")
        print(f"    Time variance  : {f['time_variance']:.1f}s")
        print(f"    Topics tried   : {int(f['topics_tried'])}")
        print(f"    Sessions done  : {int(f['session_count'])}")
        print(f"\n  Explanation:")
        print(f"    {result['explanation']}")
        print(f"{'='*52}\n")

    @staticmethod
    def _explain(personality: str, features: np.ndarray) -> str:
        avg_score, score_trend, avg_time, time_variance, topics_tried, sessions = features

        if personality == "curious":
            return (f"Student scores well ({avg_score:.0f}/100), "
                    f"is {'improving' if score_trend >= 0 else 'consistent'}, "
                    f"answers quickly ({avg_time:.0f}s), "
                    f"and explores many topics ({int(topics_tried)} tried).")
        elif personality == "lazy":
            return (f"Student has lower scores ({avg_score:.0f}/100), "
                    f"takes longer to answer ({avg_time:.0f}s), "
                    f"and has completed fewer sessions ({int(sessions)}).")
        else:  # anxious
            return (f"Student shows inconsistent timing (variance: {time_variance:.0f}s) "
                    f"and {'declining' if score_trend < 0 else 'unstable'} scores "
                    f"({avg_score:.0f}/100) — likely stressed by difficulty changes.")

    @staticmethod
    def _feature_dict(features: np.ndarray) -> dict:
        keys = ["avg_score", "score_trend", "avg_time",
                "time_variance", "topics_tried", "session_count"]
        return {k: round(float(v), 2) for k, v in zip(keys, features)}


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST — run:  python personality_detector.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    detector = PersonalityDetector()
    detector.train()

    # Test with sample student IDs from ml_engine sample data
    for student in ["s1", "s2", "priya"]:
        detector.full_report(student)
