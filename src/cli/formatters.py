from typing import Any
import json


def format_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def format_text(obj: Any) -> str:
    # Simple human-readable formatter for quick inspection
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append(str(obj))
    return "\n".join(lines)
