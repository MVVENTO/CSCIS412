from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RISK_STATEMENT = "Confidential data exposure via user prompts and retention."

SENSITIVE_PATTERNS = {
    "customer_id": re.compile(r"CUST-\d{4}"),
    "invoice_id": re.compile(r"INV-\d{4}"),
    "project_code": re.compile(r"PRJ-[A-Z]+-\d{3}"),
    "api_key": re.compile(r"sk-demo-[A-Za-z0-9-]+"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.example"),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_fake_source_frames() -> dict[str, pd.DataFrame]:
    customers = pd.DataFrame(
        [
            {
                "customer_id": "CUST-1001",
                "company_name": "Northwind Surgical",
                "contact_name": "Maya Patel",
                "contact_email": "maya.patel@northwind.example",
                "plan_tier": "Enterprise",
                "renewal_date": "2026-05-15",
            },
            {
                "customer_id": "CUST-1002",
                "company_name": "Lakeview Clinics",
                "contact_name": "Ethan Ross",
                "contact_email": "ethan.ross@lakeview.example",
                "plan_tier": "Business",
                "renewal_date": "2026-06-01",
            },
        ]
    )

    invoices = pd.DataFrame(
        [
            {
                "invoice_id": "INV-1048",
                "customer_id": "CUST-1001",
                "amount_due": 48250.00,
                "currency": "USD",
                "due_date": "2026-04-30",
                "status": "Overdue",
                "bank_reference": "ACCT-7781-FAKE",
            },
            {
                "invoice_id": "INV-1051",
                "customer_id": "CUST-1002",
                "amount_due": 19300.00,
                "currency": "USD",
                "due_date": "2026-05-08",
                "status": "Open",
                "bank_reference": "ACCT-4419-FAKE",
            },
        ]
    )

    project_notes = pd.DataFrame(
        [
            {
                "project_code": "PRJ-AURORA-221",
                "classification": "Confidential",
                "owner": "Platform Security",
                "note_text": (
                    "Project Aurora is preparing a beta rollout that migrates "
                    "8,400 patient claim records into the internal assistant sandbox."
                ),
            },
            {
                "project_code": "PRJ-DELTA-118",
                "classification": "Internal",
                "owner": "Finance Systems",
                "note_text": "Delta will replace the legacy invoice approval macros in Q3.",
            },
        ]
    )

    api_keys = pd.DataFrame(
        [
            {
                "system_name": "claims-summarizer",
                "environment": "staging",
                "api_key": "sk-demo-aurora-7F4X9Q2L8M1N",
                "owner": "A. Chen",
                "rotation_due": "2026-05-01",
            },
            {
                "system_name": "billing-exporter",
                "environment": "sandbox",
                "api_key": "sk-demo-billing-5Q1L2M8R4T7Z",
                "owner": "K. Lewis",
                "rotation_due": "2026-05-10",
            },
        ]
    )

    meeting_notes = pd.DataFrame(
        [
            {
                "meeting_id": "MEET-302",
                "meeting_date": "2026-04-17",
                "attendees": "Sales Ops, Finance, Customer Success",
                "note_text": (
                    "Finance approved a 12 percent discount for Northwind Surgical "
                    "if the renewal closes before 2026-05-15. Keep the concession internal."
                ),
            },
            {
                "meeting_id": "MEET-304",
                "meeting_date": "2026-04-18",
                "attendees": "Support, Platform Security",
                "note_text": "Do not share sandbox credentials in external support channels.",
            },
        ]
    )

    return {
        "customers": customers,
        "invoices": invoices,
        "project_notes": project_notes,
        "api_keys": api_keys,
        "meeting_notes": meeting_notes,
    }


def initialize_demo_database(db_path: Path) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                contact_name TEXT NOT NULL,
                contact_email TEXT NOT NULL,
                plan_tier TEXT NOT NULL,
                renewal_date TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS invoices (
                invoice_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                amount_due REAL NOT NULL,
                currency TEXT NOT NULL,
                due_date TEXT NOT NULL,
                status TEXT NOT NULL,
                bank_reference TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS project_notes (
                project_code TEXT PRIMARY KEY,
                classification TEXT NOT NULL,
                owner TEXT NOT NULL,
                note_text TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                system_name TEXT NOT NULL,
                environment TEXT NOT NULL,
                api_key TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                rotation_due TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meeting_notes (
                meeting_id TEXT PRIMARY KEY,
                meeting_date TEXT NOT NULL,
                attendees TEXT NOT NULL,
                note_text TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                submitted_at TEXT NOT NULL,
                user_name TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                memory_enabled INTEGER NOT NULL,
                vendor_logging_enabled INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prompt_leakage_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                source_table TEXT NOT NULL,
                source_reference TEXT NOT NULL,
                leaked_field TEXT NOT NULL,
                leaked_value TEXT NOT NULL,
                reason TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message_text TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                sensitive_matches_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS browser_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                page_title TEXT NOT NULL,
                url TEXT NOT NULL,
                snippet TEXT NOT NULL,
                stored_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS telemetry_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_name TEXT NOT NULL,
                prompt_chars INTEGER NOT NULL,
                contains_sensitive_data INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                captured_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vendor_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                retention_policy TEXT NOT NULL,
                payload_excerpt TEXT NOT NULL,
                full_prompt TEXT NOT NULL,
                captured_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
    return db_path


def seed_demo_database(db_path: Path) -> dict[str, int]:
    frames = build_fake_source_frames()
    with sqlite3.connect(db_path) as conn:
        for table_name, frame in frames.items():
            conn.execute(f"DELETE FROM {table_name}")
            frame.to_sql(table_name, conn, if_exists="append", index=False)

        for table_name in [
            "chat_sessions",
            "prompt_leakage_items",
            "chat_logs",
            "browser_history",
            "telemetry_events",
            "vendor_logs",
            "conversation_memory",
        ]:
            conn.execute(f"DELETE FROM {table_name}")

    return {table_name: len(frame) for table_name, frame in frames.items()}


def fetch_table(db_path: Path, table_name: str, order_by: str | None = None) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    if order_by:
        query += f" ORDER BY {order_by}"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def load_source_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    return {
        table_name: fetch_table(db_path, table_name)
        for table_name in [
            "customers",
            "invoices",
            "project_notes",
            "api_keys",
            "meeting_notes",
        ]
    }


def build_leaky_prompt(db_path: Path) -> tuple[str, pd.DataFrame]:
    customer = fetch_table(db_path, "customers").iloc[0]
    invoice = fetch_table(db_path, "invoices").iloc[0]
    project_note = fetch_table(db_path, "project_notes").iloc[0]
    api_key = fetch_table(db_path, "api_keys").iloc[0]
    meeting_note = fetch_table(db_path, "meeting_notes").iloc[0]

    leaked_items = pd.DataFrame(
        [
            {
                "source_table": "customers",
                "source_reference": customer["customer_id"],
                "leaked_field": "contact_email",
                "leaked_value": customer["contact_email"],
                "reason": "Personally identifying customer contact information.",
            },
            {
                "source_table": "invoices",
                "source_reference": invoice["invoice_id"],
                "leaked_field": "amount_due",
                "leaked_value": f"{invoice['currency']} {invoice['amount_due']:,.2f}",
                "reason": "Non-public financial balance and invoice reference.",
            },
            {
                "source_table": "project_notes",
                "source_reference": project_note["project_code"],
                "leaked_field": "note_text",
                "leaked_value": project_note["note_text"],
                "reason": "Confidential internal project details.",
            },
            {
                "source_table": "api_keys",
                "source_reference": api_key["system_name"],
                "leaked_field": "api_key",
                "leaked_value": api_key["api_key"],
                "reason": "Credential material that should never be pasted into an AI tool.",
            },
            {
                "source_table": "meeting_notes",
                "source_reference": meeting_note["meeting_id"],
                "leaked_field": "note_text",
                "leaked_value": meeting_note["note_text"],
                "reason": "Internal pricing strategy and discount concessions.",
            },
        ]
    )

    prompt = f"""Draft an executive update for tomorrow's renewal call and summarize the exposure risk.

Customer context:
- Customer ID: {customer['customer_id']}
- Company: {customer['company_name']}
- Contact: {customer['contact_name']}
- Contact email: {customer['contact_email']}
- Renewal date: {customer['renewal_date']}

Billing context:
- Invoice: {invoice['invoice_id']}
- Amount due: {invoice['currency']} {invoice['amount_due']:,.2f}
- Due date: {invoice['due_date']}
- Bank reference: {invoice['bank_reference']}

Confidential project note:
- {project_note['project_code']} ({project_note['classification']}): {project_note['note_text']}

Temporary API credential:
- {api_key['system_name']} {api_key['environment']} key: {api_key['api_key']}

Meeting note:
- {meeting_note['note_text']}

Please turn this into a concise customer-facing summary and tell me what issues stand out."""

    return prompt, leaked_items


def detect_sensitive_matches(text: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for label, pattern in SENSITIVE_PATTERNS.items():
        for match in pattern.finditer(text):
            hits.append({"label": label, "match": match.group(0)})
    return hits


def summarize_sensitive_matches(text: str) -> dict[str, Any]:
    hits = detect_sensitive_matches(text)
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit["label"]] = counts.get(hit["label"], 0) + 1
    return {"total_matches": len(hits), "counts": counts, "hits": hits}


def build_mock_ai_response(prompt: str) -> str:
    return (
        "I drafted a summary for Northwind Surgical, referenced invoice INV-1048, "
        "noted the overdue balance, and included the Aurora rollout context and "
        "discount note. The pasted prompt contains confidential data that should be removed."
    )


def simulate_accidental_leakage(
    db_path: Path,
    *,
    memory_enabled: bool = True,
    vendor_logging_enabled: bool = True,
    user_name: str = "Casey Morgan",
    tool_name: str = "Internal Productivity Copilot",
) -> dict[str, Any]:
    prompt, leaked_items = build_leaky_prompt(db_path)
    response = build_mock_ai_response(prompt)
    prompt_summary = summarize_sensitive_matches(prompt)
    response_summary = summarize_sensitive_matches(response)
    session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    created_at = utc_now_iso()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (
                session_id, submitted_at, user_name, tool_name,
                memory_enabled, vendor_logging_enabled
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                created_at,
                user_name,
                tool_name,
                int(memory_enabled),
                int(vendor_logging_enabled),
            ),
        )

        for row in leaked_items.to_dict(orient="records"):
            conn.execute(
                """
                INSERT INTO prompt_leakage_items (
                    session_id, source_table, source_reference,
                    leaked_field, leaked_value, reason
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    row["source_table"],
                    row["source_reference"],
                    row["leaked_field"],
                    row["leaked_value"],
                    row["reason"],
                ),
            )

        conn.execute(
            """
            INSERT INTO chat_logs (
                session_id, role, message_text, token_estimate,
                sensitive_matches_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                "user",
                prompt,
                max(1, len(prompt) // 4),
                json.dumps(prompt_summary),
                created_at,
            ),
        )
        conn.execute(
            """
            INSERT INTO chat_logs (
                session_id, role, message_text, token_estimate,
                sensitive_matches_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                "assistant",
                response,
                max(1, len(response) // 4),
                json.dumps(response_summary),
                created_at,
            ),
        )

        conn.execute(
            """
            INSERT INTO browser_history (
                session_id, page_title, url, snippet, stored_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                "Internal Productivity Copilot Conversation",
                f"https://copilot.example.internal/chat/{session_id}",
                prompt[:180],
                created_at,
            ),
        )

        telemetry_metadata = {
            "prompt_preview": prompt[:140],
            "prompt_length": len(prompt),
            "sensitive_counts": prompt_summary["counts"],
        }
        conn.execute(
            """
            INSERT INTO telemetry_events (
                session_id, event_name, prompt_chars,
                contains_sensitive_data, metadata_json, captured_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                "prompt_submitted",
                len(prompt),
                int(prompt_summary["total_matches"] > 0),
                json.dumps(telemetry_metadata),
                created_at,
            ),
        )

        if vendor_logging_enabled:
            conn.execute(
                """
                INSERT INTO vendor_logs (
                    session_id, provider_name, retention_policy,
                    payload_excerpt, full_prompt, captured_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    "External LLM Vendor",
                    "Retained for support and abuse monitoring in this demo.",
                    prompt[:160],
                    prompt,
                    created_at,
                ),
            )

        if memory_enabled:
            memory_rows = [
                (
                    session_id,
                    "customer_context",
                    "Northwind Surgical has an enterprise renewal with an overdue invoice.",
                    created_at,
                ),
                (
                    session_id,
                    "project_context",
                    "Project Aurora involves internal migration details and should stay confidential.",
                    created_at,
                ),
                (
                    session_id,
                    "pricing_context",
                    "A 12 percent discount concession was discussed for the renewal.",
                    created_at,
                ),
            ]
            conn.executemany(
                """
                INSERT INTO conversation_memory (
                    session_id, memory_key, memory_value, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                memory_rows,
            )

    return {
        "session_id": session_id,
        "prompt": prompt,
        "response": response,
        "memory_enabled": memory_enabled,
        "vendor_logging_enabled": vendor_logging_enabled,
        "prompt_summary": prompt_summary,
        "shared_items": leaked_items,
    }


def fetch_session_artifacts(db_path: Path, session_id: str) -> dict[str, pd.DataFrame]:
    artifacts: dict[str, pd.DataFrame] = {}
    query_map = {
        "chat_logs": "SELECT role, message_text, sensitive_matches_json, created_at "
        "FROM chat_logs WHERE session_id = ? ORDER BY id",
        "browser_history": "SELECT page_title, url, snippet, stored_at "
        "FROM browser_history WHERE session_id = ? ORDER BY id",
        "telemetry_events": "SELECT event_name, prompt_chars, contains_sensitive_data, "
        "metadata_json, captured_at FROM telemetry_events WHERE session_id = ? ORDER BY id",
        "vendor_logs": "SELECT provider_name, retention_policy, payload_excerpt, captured_at "
        "FROM vendor_logs WHERE session_id = ? ORDER BY id",
        "conversation_memory": "SELECT memory_key, memory_value, created_at "
        "FROM conversation_memory WHERE session_id = ? ORDER BY id",
        "prompt_leakage_items": "SELECT source_table, source_reference, leaked_field, leaked_value, reason "
        "FROM prompt_leakage_items WHERE session_id = ? ORDER BY id",
    }

    with sqlite3.connect(db_path) as conn:
        for name, query in query_map.items():
            artifacts[name] = pd.read_sql_query(query, conn, params=(session_id,))

    return artifacts


def build_storage_summary(db_path: Path, session_id: str) -> pd.DataFrame:
    artifacts = fetch_session_artifacts(db_path, session_id)
    rows: list[dict[str, str]] = []

    if not artifacts["chat_logs"].empty:
        rows.append(
            {
                "location": "Chat logs",
                "stored_data": artifacts["chat_logs"].iloc[0]["message_text"][:120] + "...",
                "why_it_exists": "Conversation history and support review.",
                "risk": "Full prompt retention preserves all pasted confidential text.",
            }
        )

    if not artifacts["browser_history"].empty:
        rows.append(
            {
                "location": "Browser history",
                "stored_data": artifacts["browser_history"].iloc[0]["snippet"][:120] + "...",
                "why_it_exists": "Session restore, autocomplete, and synced history.",
                "risk": "Prompt fragments can remain on the local device or synced profile.",
            }
        )

    if not artifacts["telemetry_events"].empty:
        rows.append(
            {
                "location": "Telemetry",
                "stored_data": artifacts["telemetry_events"].iloc[0]["metadata_json"][:120] + "...",
                "why_it_exists": "Analytics, debugging, and observability.",
                "risk": "Prompt previews and sensitive counts leak into analytics pipelines.",
            }
        )

    if not artifacts["vendor_logs"].empty:
        rows.append(
            {
                "location": "Vendor logs",
                "stored_data": artifacts["vendor_logs"].iloc[0]["payload_excerpt"][:120] + "...",
                "why_it_exists": "External provider support and abuse monitoring.",
                "risk": "Confidential prompts leave the company boundary.",
            }
        )

    if not artifacts["conversation_memory"].empty:
        rows.append(
            {
                "location": "Conversation memory",
                "stored_data": artifacts["conversation_memory"].iloc[0]["memory_value"],
                "why_it_exists": "Future personalization and context reuse.",
                "risk": "Sensitive facts can resurface in later prompts or sessions.",
            }
        )

    return pd.DataFrame(rows)


def render_flow_diagram(
    *,
    tool_name: str = "Internal Productivity Copilot",
    memory_enabled: bool = True,
    vendor_logging_enabled: bool = True,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def add_box(x: float, y: float, text: str, color: str) -> None:
        box = FancyBboxPatch(
            (x, y),
            2.5,
            1.0,
            boxstyle="round,pad=0.2",
            facecolor=color,
            edgecolor="#1f2937",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x + 1.25, y + 0.5, text, ha="center", va="center", fontsize=10, weight="bold")

    def add_arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.6,
                color="#374151",
            )
        )

    add_box(0.5, 2.4, "Employee\npastes prompt", "#dbeafe")
    add_box(4.0, 2.4, tool_name, "#fde68a")
    add_box(8.0, 4.3, "Chat logs", "#fecaca")
    add_box(8.0, 2.8, "Browser history", "#fecaca")
    add_box(8.0, 1.3, "Telemetry", "#fecaca")

    if vendor_logging_enabled:
        add_box(10.3, 4.3, "Vendor logs", "#fca5a5")
    if memory_enabled:
        add_box(10.3, 1.3, "Conversation\nmemory", "#fca5a5")

    add_arrow((3.0, 2.9), (4.0, 2.9))
    add_arrow((6.5, 3.1), (8.0, 4.8))
    add_arrow((6.5, 2.9), (8.0, 3.3))
    add_arrow((6.5, 2.7), (8.0, 1.8))

    if vendor_logging_enabled:
        add_arrow((10.5, 4.8), (10.3, 4.8))
    if memory_enabled:
        add_arrow((10.5, 1.8), (10.3, 1.8))

    ax.set_title("Accidental Data Leakage Path: user prompt to retained artifacts", fontsize=14, weight="bold")
    return fig
