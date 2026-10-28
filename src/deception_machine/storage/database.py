"""
SQLite database for experiment storage.

Stores complete experimental records for reproducibility and analysis.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models import (
    ExperimentConfig,
    ExperimentSummary,
    TrialRecord,
    ExperimentalCondition,
    ReportClassification,
)


class ExperimentDatabase:
    """SQLite database for storing experiment data."""

    def __init__(self, db_path: str = "experiments.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT DEFAULT 'running'
            );

            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                condition TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                random_seed INTEGER NOT NULL,

                -- Project data
                project_metrics_json TEXT NOT NULL,
                incentive_json TEXT NOT NULL,

                -- Agent output
                report_content TEXT NOT NULL,
                report_raw TEXT NOT NULL,
                executive_report_content TEXT,
                executive_report_raw TEXT,

                -- Evaluation
                factual_accuracy REAL NOT NULL,
                omission_score REAL NOT NULL,
                misrepresentation_rate REAL NOT NULL,
                classification TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,

                -- Executive evaluation (for condition E)
                executive_evaluation_json TEXT,

                -- Model config
                model_name TEXT NOT NULL,
                temperature REAL NOT NULL,
                prompt_used TEXT NOT NULL,

                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            );

            CREATE INDEX IF NOT EXISTS idx_trials_experiment
                ON trials(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_trials_condition
                ON trials(condition);
            CREATE INDEX IF NOT EXISTS idx_trials_classification
                ON trials(classification);
        """)
        conn.commit()

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment record."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO experiments (experiment_id, name, config_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                config.experiment_id,
                config.name,
                config.model_dump_json(),
                datetime.now().isoformat(),
            )
        )
        conn.commit()
        return config.experiment_id

    def complete_experiment(self, experiment_id: str) -> None:
        """Mark experiment as completed."""
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE experiments
            SET completed_at = ?, status = 'completed'
            WHERE experiment_id = ?
            """,
            (datetime.now().isoformat(), experiment_id)
        )
        conn.commit()

    def save_trial(self, trial: TrialRecord) -> None:
        """Save a trial record."""
        conn = self._get_conn()

        executive_content = None
        executive_raw = None
        executive_eval = None

        if trial.executive_report:
            executive_content = trial.executive_report.content
            executive_raw = trial.executive_report.raw_response
        if trial.executive_evaluation:
            executive_eval = trial.executive_evaluation.model_dump_json()

        conn.execute(
            """
            INSERT INTO trials (
                trial_id, experiment_id, condition, trial_number, timestamp,
                random_seed, project_metrics_json, incentive_json,
                report_content, report_raw, executive_report_content,
                executive_report_raw, factual_accuracy, omission_score,
                misrepresentation_rate, classification, evaluation_json,
                executive_evaluation_json, model_name, temperature, prompt_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trial.trial_id,
                trial.experiment_id,
                trial.condition.value,
                trial.trial_number,
                trial.timestamp.isoformat(),
                trial.random_seed,
                trial.project_metrics.model_dump_json(),
                trial.incentive_structure.model_dump_json(),
                trial.agent_report.content,
                trial.agent_report.raw_response,
                executive_content,
                executive_raw,
                trial.evaluation.factual_accuracy_score,
                trial.evaluation.material_omission_score,
                trial.evaluation.misrepresentation_rate,
                trial.evaluation.classification.value,
                trial.evaluation.model_dump_json(),
                executive_eval,
                trial.agent_report.model_name,
                trial.agent_report.temperature,
                trial.agent_report.prompt_used,
            )
        )
        conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        """Get experiment metadata."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,)
        ).fetchone()

        if row is None:
            return None

        return {
            "experiment_id": row["experiment_id"],
            "name": row["name"],
            "config": json.loads(row["config_json"]),
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "status": row["status"],
        }

    def get_trials(
        self,
        experiment_id: str,
        condition: Optional[ExperimentalCondition] = None,
    ) -> list[dict]:
        """Get trials for an experiment."""
        conn = self._get_conn()

        if condition:
            rows = conn.execute(
                """
                SELECT * FROM trials
                WHERE experiment_id = ? AND condition = ?
                ORDER BY trial_number
                """,
                (experiment_id, condition.value)
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM trials
                WHERE experiment_id = ?
                ORDER BY condition, trial_number
                """,
                (experiment_id,)
            ).fetchall()

        return [dict(row) for row in rows]

    def list_experiments(self) -> list[dict]:
        """List all experiments."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT e.*, COUNT(t.trial_id) as trial_count
            FROM experiments e
            LEFT JOIN trials t ON e.experiment_id = t.experiment_id
            GROUP BY e.experiment_id
            ORDER BY e.created_at DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]

    def get_summary_stats(self, experiment_id: str) -> dict:
        """Get summary statistics for an experiment."""
        conn = self._get_conn()

        # Overall stats
        overall = conn.execute(
            """
            SELECT
                COUNT(*) as total_trials,
                AVG(factual_accuracy) as mean_accuracy,
                AVG(omission_score) as mean_omission_score,
                AVG(misrepresentation_rate) as mean_misrep_rate
            FROM trials
            WHERE experiment_id = ?
            """,
            (experiment_id,)
        ).fetchone()

        # Per-condition stats
        by_condition = conn.execute(
            """
            SELECT
                condition,
                COUNT(*) as trials,
                AVG(factual_accuracy) as mean_accuracy,
                AVG(omission_score) as mean_omission_score,
                AVG(misrepresentation_rate) as mean_misrep_rate,
                MIN(factual_accuracy) as min_accuracy,
                MAX(factual_accuracy) as max_accuracy
            FROM trials
            WHERE experiment_id = ?
            GROUP BY condition
            ORDER BY condition
            """,
            (experiment_id,)
        ).fetchall()

        # Classification distribution
        classifications = conn.execute(
            """
            SELECT condition, classification, COUNT(*) as count
            FROM trials
            WHERE experiment_id = ?
            GROUP BY condition, classification
            """,
            (experiment_id,)
        ).fetchall()

        # Organize classification data
        class_dist = {}
        for row in classifications:
            cond = row["condition"]
            if cond not in class_dist:
                class_dist[cond] = {}
            class_dist[cond][row["classification"]] = row["count"]

        return {
            "total_trials": overall["total_trials"],
            "mean_accuracy": overall["mean_accuracy"],
            "mean_omission_score": overall["mean_omission_score"],
            "mean_misrepresentation_rate": overall["mean_misrep_rate"],
            "by_condition": [dict(row) for row in by_condition],
            "classification_distribution": class_dist,
        }

    def get_audit_stats(self, experiment_id: str) -> dict:
        """Get audit-related statistics for condition D."""
        conn = self._get_conn()

        rows = conn.execute(
            """
            SELECT evaluation_json
            FROM trials
            WHERE experiment_id = ? AND condition = ?
            """,
            (experiment_id, ExperimentalCondition.D_AUDIT_AWARENESS.value)
        ).fetchall()

        total = len(rows)
        audited = 0
        detected = 0

        for row in rows:
            eval_data = json.loads(row["evaluation_json"])
            if eval_data.get("audit_triggered"):
                audited += 1
                if eval_data.get("audit_detected_issues"):
                    detected += 1

        return {
            "total_trials": total,
            "audits_triggered": audited,
            "issues_detected": detected,
            "audit_rate": audited / total if total > 0 else 0,
            "detection_rate": detected / audited if audited > 0 else 0,
        }

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment and its trials."""
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM trials WHERE experiment_id = ?",
            (experiment_id,)
        )
        conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?",
            (experiment_id,)
        )
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
