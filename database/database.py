import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("database")

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inspections.db")

class InspectionDatabase:
    """Manages SQLite storage and historical query retrieval for inspection operations."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a SQLite connection with dict-like row factory and pragmas enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable Write-Ahead Logging for improved concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_database(self) -> None:
        """Creates the inspection table and indices if they do not already exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    result TEXT NOT NULL CHECK(result IN ('PASS', 'FAIL')),
                    measured_width REAL NOT NULL,
                    measured_height REAL NOT NULL,
                    measured_area REAL NOT NULL,
                    measured_perimeter REAL NOT NULL,
                    circularity REAL NOT NULL,
                    defects_count INTEGER NOT NULL DEFAULT 0,
                    defects_details TEXT DEFAULT '[]',
                    defect_summary TEXT DEFAULT 'None',
                    processing_time_ms REAL NOT NULL,
                    source TEXT DEFAULT 'Manual',
                    config_summary TEXT DEFAULT '{}'
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_result ON inspections(result);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_product_id ON inspections(product_id);")
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def save_inspection(
        self,
        product_id: str,
        result: str,
        measured_width: float,
        measured_height: float,
        measured_area: float,
        measured_perimeter: float,
        circularity: float,
        defects_count: int,
        defects_details: List[Dict[str, Any]],
        defect_summary: str,
        processing_time_ms: float,
        source: str = "Upload",
        config_summary: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ) -> int:
        """Inserts an inspection record and returns the new row ID."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        details_json = json.dumps(defects_details)
        config_json = json.dumps(config_summary or {})

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO inspections (
                        product_id, timestamp, result,
                        measured_width, measured_height, measured_area, measured_perimeter, circularity,
                        defects_count, defects_details, defect_summary,
                        processing_time_ms, source, config_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product_id, timestamp, result,
                    round(float(measured_width), 2),
                    round(float(measured_height), 2),
                    round(float(measured_area), 2),
                    round(float(measured_perimeter), 2),
                    round(float(circularity), 4),
                    int(defects_count),
                    details_json,
                    defect_summary,
                    round(float(processing_time_ms), 2),
                    source,
                    config_json
                ))
                record_id = cursor.lastrowid
                logger.info(f"Saved inspection #{record_id} for Product {product_id} with status: {result}")
                return record_id
        finally:
            conn.close()

    def get_history(self, limit: int = 500) -> pd.DataFrame:
        """Retrieves historical inspection records as a formatted pandas DataFrame."""
        conn = self._get_connection()
        try:
            query = f"""
                SELECT 
                    id AS "ID",
                    product_id AS "Product ID",
                    timestamp AS "Timestamp",
                    result AS "Result",
                    measured_width AS "Width (px)",
                    measured_height AS "Height (px)",
                    measured_area AS "Area (px²)",
                    measured_perimeter AS "Perimeter (px)",
                    circularity AS "Circularity",
                    defects_count AS "Defect Count",
                    defect_summary AS "Defects",
                    processing_time_ms AS "Time (ms)",
                    source AS "Source"
                FROM inspections
                ORDER BY id DESC
                LIMIT {int(limit)};
            """
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates aggregate quality metrics across all recorded inspections."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(processing_time_ms) FROM inspections;")
            total_row = cursor.fetchone()
            total_count = total_row[0] or 0
            avg_time = round(total_row[1] or 0.0, 2)

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE result = 'PASS';")
            pass_count = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM inspections WHERE result = 'FAIL';")
            fail_count = cursor.fetchone()[0] or 0

            pass_rate = round((pass_count / total_count * 100), 2) if total_count > 0 else 0.0

            cursor.execute("SELECT defects_details FROM inspections WHERE defects_count > 0;")
            rows = cursor.fetchall()

            defect_type_counts: Dict[str, int] = {}
            for r in rows:
                try:
                    defects = json.loads(r[0])
                    for d in defects:
                        dtype = d.get("type", "Unknown Defect")
                        defect_type_counts[dtype] = defect_type_counts.get(dtype, 0) + 1
                except Exception:
                    pass

            cursor.execute("SELECT defect_summary FROM inspections WHERE result = 'FAIL';")
            fail_summaries = cursor.fetchall()
            for s in fail_summaries:
                summary_text = s[0] or ""
                if "Dimension" in summary_text or "Width" in summary_text or "Height" in summary_text:
                    defect_type_counts["Dimension Out of Spec"] = defect_type_counts.get("Dimension Out of Spec", 0) + 1

            most_common = "None"
            if defect_type_counts:
                most_common = max(defect_type_counts.items(), key=lambda x: x[1])[0]

            return {
                "total_inspected": total_count,
                "passed": pass_count,
                "failed": fail_count,
                "pass_percentage": pass_rate,
                "avg_processing_time_ms": avg_time,
                "most_common_defect": most_common,
                "defect_type_counts": defect_type_counts
            }
        finally:
            conn.close()

    def clear_history(self) -> None:
        """Deletes all records from the inspections table."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM inspections;")
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='inspections';")
            logger.warning("All inspection database history cleared.")
        finally:
            conn.close()
