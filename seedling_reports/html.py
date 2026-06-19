from __future__ import annotations

from html import escape
from pathlib import Path


def write_experiment_html_report(
    title: str,
    output_path: str | Path,
    sections: list[tuple[str, list[dict[str, object]]]],
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(_table(section_title, rows) for section_title, rows in sections)
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2528; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 13px; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f2f2ed; }}
    code {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  {body}
</body>
</html>
""",
        encoding="utf-8",
    )


def _table(title: str, rows: list[dict[str, object]]) -> str:
    if not rows:
        return f"<section><h2>{escape(title)}</h2><p>No rows.</p></section>"
    columns = list(rows[0].keys())
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"<section><h2>{escape(title)}</h2><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></section>"
