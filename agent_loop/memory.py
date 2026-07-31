"""Simple JSONL-based memory store for agent reflections.

Each entry is a line of JSON with: id, timestamp, entry, tags, turn.
Stored at ./agent_memory.jsonl by default (overridable via AGENT_MEMORY_PATH).
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_PATH = os.environ.get("AGENT_MEMORY_PATH", "./agent_memory.jsonl")


class MemoryStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_PATH

    def append(self, entry: str, tags: Optional[List[str]] = None,
               turn: Optional[int] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Append a memory entry and return it."""
        record = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entry": entry,
            "tags": tags or [],
            "turn": turn,
            "metadata": metadata or {}
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read(self, limit: int = 10, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read recent entries, optionally filtered by tag."""
        entries = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if tag and tag not in record.get("tags", []):
                            continue
                        entries.append(record)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return list(reversed(entries))[:limit]

    def search(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search entries containing a keyword (case-insensitive)."""
        entries = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if keyword.lower() in record.get("entry", "").lower():
                            entries.append(record)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return list(reversed(entries))[:limit]
