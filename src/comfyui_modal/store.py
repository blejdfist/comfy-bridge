"""SQLite is the local source of truth; remote ComfyUI queues are ephemeral."""

import json
import sqlite3
import time
from pathlib import Path

from .helpers.comfy_history import last_execution_error

TERMINAL = {"completed", "failed", "cancelled"}
ACTIVE = {"submitting", "running", "unknown", "downloading", "cancelling"}
DOWNLOAD_JSON_KEYS = ("request", "progress", "result")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            number INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
            state TEXT NOT NULL, priority REAL NOT NULL, created REAL NOT NULL,
            updated REAL NOT NULL, payload TEXT NOT NULL, call_id TEXT,
            result TEXT, history TEXT, error TEXT, event_seq INTEGER NOT NULL DEFAULT 0
        )""")
        self.db.commit()
        self.db.execute("""CREATE TABLE IF NOT EXISTS model_downloads (
            id TEXT PRIMARY KEY, state TEXT NOT NULL, created REAL NOT NULL,
            updated REAL NOT NULL, request TEXT NOT NULL, call_id TEXT,
            progress TEXT, result TEXT, error TEXT
        )""")
        self.db.commit()

    @staticmethod
    def decode(row, json_keys=("payload", "result", "history")):
        if row is None:
            return None
        value = dict(row)
        for key in json_keys:
            if value[key] is not None:
                value[key] = json.loads(value[key])
        return value

    def _update(self, table, row_id, allowed, json_keys, error, fields):
        if fields.keys() - allowed:
            raise ValueError(error)
        fields["updated"] = time.time()
        for key in json_keys:
            if key in fields:
                fields[key] = json.dumps(fields[key])
        assignment = ", ".join(f"{key}=?" for key in fields)
        self.db.execute(f"UPDATE {table} SET {assignment} WHERE id=?", (*fields.values(), row_id))
        self.db.commit()

    def get(self, job_id):
        return self.decode(self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def all(self):
        return [self.decode(row) for row in self.db.execute("SELECT * FROM jobs ORDER BY number")]

    def add(self, job_id, payload, front=False):
        now = time.time()
        minimum = self.db.execute("SELECT COALESCE(MIN(priority),0) FROM jobs").fetchone()[0]
        priority = minimum - 1 if front else now
        self.db.execute(
            """INSERT INTO jobs(id,state,priority,created,updated,payload)
                           VALUES(?, 'pending', ?, ?, ?, ?)""",
            (job_id, priority, now, now, json.dumps(payload)),
        )
        self.db.commit()
        return self.get(job_id)

    def update(self, job_id, **fields):
        allowed = {"state", "call_id", "result", "history", "error", "event_seq"}
        self._update("jobs", job_id, allowed, ("result", "history"), "Unknown job fields", fields)

    def delete(self, job_id):
        row = self.get(job_id)
        if row and row["state"] in TERMINAL:
            self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self.db.commit()

    def queue_item(self, job):
        payload = job["payload"]
        extra = dict(payload.get("extra_data", {}))
        extra["client_id"] = payload.get("client_id")
        extra["create_time"] = int(job["created"] * 1000)
        return [
            job["priority"],
            job["id"],
            payload["prompt"],
            extra,
            payload.get("outputs_to_execute", []),
        ]

    def queue(self):
        jobs = self.all()
        return {
            "queue_running": [self.queue_item(j) for j in jobs if j["state"] in ACTIVE],
            "queue_pending": [
                self.queue_item(j)
                for j in sorted(jobs, key=lambda j: j["priority"])
                if j["state"] == "pending"
            ],
        }

    def history(self, job_id=None):
        return {
            j["id"]: j["history"]
            for j in self.all()
            if j["history"] and (job_id is None or job_id == j["id"])
        }

    def close(self):
        self.db.close()

    def model_downloads(self):
        rows = self.db.execute("SELECT * FROM model_downloads ORDER BY created")
        return [self.decode(row, DOWNLOAD_JSON_KEYS) for row in rows]

    def model_download(self, download_id):
        row = self.db.execute("SELECT * FROM model_downloads WHERE id=?", (download_id,)).fetchone()
        return self.decode(row, DOWNLOAD_JSON_KEYS)

    def add_model_download(self, download_id, request):
        now = time.time()
        self.db.execute(
            """INSERT INTO model_downloads(id,state,created,updated,request)
                           VALUES(?, 'pending', ?, ?, ?)""",
            (download_id, now, now, json.dumps(request)),
        )
        self.db.commit()

    def update_model_download(self, download_id, **fields):
        allowed = {"state", "call_id", "progress", "result", "error"}
        self._update(
            "model_downloads",
            download_id,
            allowed,
            ("progress", "result"),
            "Unknown download fields",
            fields,
        )


def execution_failure(history):
    """Keep ComfyUI's actionable error and original node/traceback instead of serializing status."""
    failure = last_execution_error((history or {}).get("status", {}).get("messages", []))
    return dict(failure) if failure is not None else None


def public_job(job, include_outputs=False):
    history = job["history"] or {}
    outputs = history.get("outputs", {})
    images = [
        (node, item)
        for node, output in outputs.items()
        for item in output.get("images", [])
        if isinstance(item, dict)
    ]
    extra = job["payload"].get("extra_data", {})
    workflow_id = extra.get("extra_pnginfo", {}).get("workflow", {}).get("id")
    result = {
        "id": job["id"],
        "status": job["state"]
        if job["state"] in TERMINAL
        else "pending"
        if job["state"] == "pending"
        else "in_progress",
        "priority": job["priority"],
        "create_time": int(job["created"] * 1000),
        "outputs_count": len(images),
        "previewable_outputs_count": len(images),
    }
    # The editor validates /jobs with zod and drops the whole list if any entry is invalid.
    if workflow_id:
        result["workflow_id"] = workflow_id
    if images:
        node, image = images[0]
        result["preview_output"] = {**image, "nodeId": node, "mediaType": "images"}
    if job["state"] in TERMINAL:
        result["execution_end_time"] = int(job["updated"] * 1000)
    if job["error"]:
        result["execution_error"] = execution_failure(history) or {
            "exception_message": job["error"],
            "exception_type": "RemoteExecution",
            "node_id": "",
            "node_type": "",
            "traceback": [],
            "current_inputs": {},
            "current_outputs": {},
        }
    if include_outputs:
        result.update(
            outputs=outputs,
            execution_status=history.get("status", {}),
            workflow={"prompt": job["payload"]["prompt"], "extra_data": extra},
        )
    return result
