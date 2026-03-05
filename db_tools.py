

# db_tools.py
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool

load_dotenv(override=True)

DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")
DB_DRIVER   = os.getenv("DB_DRIVER", "mysqlconnector")

# ── Shared engine ──────────────────────────────────────────────────────────────
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        db_uri = f"mysql+{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        _engine = create_engine(db_uri)
    return _engine


# ── Standard SQL + REPL tools ──────────────────────────────────────────────────

def make_sql_tools(llm):
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        raise ValueError("DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME must all be set in .env")
    db_uri = f"mysql+{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    db = SQLDatabase.from_uri(db_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    all_tools = toolkit.get_tools()
    return [t for t in all_tools if t.name != "sql_db_query_checker"]

def make_python_repl_tool():
    return PythonREPLTool()


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _fmt(val):
    try:
        return f"{float(val):.2f}" if val is not None else "N/A"
    except Exception:
        return "N/A"

def _safe_table_name(device_id: str) -> str:
    """Sanitize device_id for use as a MySQL table name (backtick-quoted)."""
    # Strip or replace characters that break unquoted table names
    safe = device_id.strip()
    return f"`sensor_data_{safe}`"

def _sensor_query(device_id: str, all_time: bool, hours: int) -> str:
    table = _safe_table_name(device_id)
    time_clause = "" if all_time else f"WHERE timestamp >= NOW() - INTERVAL {int(hours)} HOUR"
    return f"""
        SELECT
            MIN(temperature_one) AS t1_min, MAX(temperature_one) AS t1_max,
            AVG(temperature_one) AS t1_avg, STDDEV(temperature_one) AS t1_std,
            MIN(temperature_two) AS t2_min, MAX(temperature_two) AS t2_max,
            AVG(temperature_two) AS t2_avg, STDDEV(temperature_two) AS t2_std,
            MIN(vibration_x)     AS vx_min, MAX(vibration_x)     AS vx_max,
            AVG(vibration_x)     AS vx_avg, STDDEV(vibration_x)  AS vx_std,
            MIN(vibration_y)     AS vy_min, MAX(vibration_y)     AS vy_max,
            AVG(vibration_y)     AS vy_avg, STDDEV(vibration_y)  AS vy_std,
            MIN(vibration_z)     AS vz_min, MAX(vibration_z)     AS vz_max,
            AVG(vibration_z)     AS vz_avg, STDDEV(vibration_z)  AS vz_std
        FROM {table}
        {time_clause}
    """

def _format_block(s, device_id: str, meta_line: str, label: str) -> str:
    return (
        f"Device: {device_id}\n{meta_line}\n\n"
        f"Sensor Summary ({label}):\n"
        f"  Temperature One -> min={_fmt(s.t1_min)}, max={_fmt(s.t1_max)}, avg={_fmt(s.t1_avg)}, std={_fmt(s.t1_std)}\n"
        f"  Temperature Two -> min={_fmt(s.t2_min)}, max={_fmt(s.t2_max)}, avg={_fmt(s.t2_avg)}, std={_fmt(s.t2_std)}\n"
        f"  Vibration X     -> min={_fmt(s.vx_min)}, max={_fmt(s.vx_max)}, avg={_fmt(s.vx_avg)}, std={_fmt(s.vx_std)}\n"
        f"  Vibration Y     -> min={_fmt(s.vy_min)}, max={_fmt(s.vy_max)}, avg={_fmt(s.vy_avg)}, std={_fmt(s.vy_std)}\n"
        f"  Vibration Z     -> min={_fmt(s.vz_min)}, max={_fmt(s.vz_max)}, avg={_fmt(s.vz_avg)}, std={_fmt(s.vz_std)}"
    ).strip()

# Full 4-table JOIN used everywhere: company -> workshops -> assets -> sensor
_BASE_JOIN = """
    FROM assets a
    JOIN workshops w  ON a.workshop_id_fk = w.workshop_id
    JOIN Company c    ON w.company_id_fk  = c.company_id
"""

def _fetch_meta(device_id: str) -> str:
    engine = get_engine()
    q = f"""
        SELECT a.asset_name, a.asset_type, a.site, a.application,
               w.workshop_name, w.workshop_address,
               c.company_id, c.company_name, c.company_address
        {_BASE_JOIN}
        WHERE REPLACE(a.asset_name, '-', '') = %s
        LIMIT 1
    """
    try:
        df = pd.read_sql(q, engine, params=(device_id,))
    except Exception:
        return "Metadata not found."
    if df.empty:
        return "Metadata not found."
    m = df.iloc[0]
    return (
        f"Asset: {m.get('asset_name','N/A')} | Type: {m.get('asset_type','N/A')} | "
        f"Site: {m.get('site','N/A')} | Application: {m.get('application','N/A')} | "
        f"Workshop: {m.get('workshop_name','N/A')} ({m.get('workshop_address','N/A')}) | "
        f"Company: {m.get('company_name','N/A')} ({m.get('company_address','N/A')})"
    )

def _resolve_devices(filter_value: str) -> pd.DataFrame:
    """
    Resolve any filter value against ALL four tables:
    company, workshops, assets columns.
    Returns DataFrame with device_id + metadata.
    """
    engine = get_engine()

    company_cols  = ["company_name", "company_address", "company_description", "company_email", "company_mobile"]
    workshop_cols = ["workshop_name", "workshop_address", "workshop_description"]
    asset_cols    = ["site", "application", "asset_type", "class", "asset_description"]

    conditions, params = [], []
    for col in company_cols:
        conditions.append(f"LOWER(c.{col}) = LOWER(%s)")
        params.append(filter_value)
    for col in workshop_cols:
        conditions.append(f"LOWER(w.{col}) = LOWER(%s)")
        params.append(filter_value)
    for col in asset_cols:
        conditions.append(f"LOWER(a.{col}) = LOWER(%s)")
        params.append(filter_value)

    q = f"""
        SELECT REPLACE(a.asset_name, '-', '') AS device_id, a.asset_name,
               a.site, a.application, a.asset_type,
               w.workshop_name, w.workshop_address,
               c.company_name, c.company_address
        {_BASE_JOIN}
        WHERE {" OR ".join(conditions)}
    """
    try:
        return pd.read_sql(q, engine, params=tuple(params))
    except Exception:
        return pd.DataFrame()

def _fetch_all_devices() -> pd.DataFrame:
    """Fetch every device across all companies/workshops."""
    engine = get_engine()
    q = f"""
        SELECT REPLACE(a.asset_name, '-', '') AS device_id, a.asset_name,
               a.site, a.application, a.asset_type,
               w.workshop_name, w.workshop_address,
               c.company_name, c.company_address
        {_BASE_JOIN}
        ORDER BY c.company_name, w.workshop_name, a.asset_name
    """
    try:
        return pd.read_sql(q, engine)
    except Exception:
        return pd.DataFrame()


# ── Pre-built fast tools 

@tool
def get_device_status(device_id: str, hours: int = 24, all_time: bool = False) -> str:
    """
    Get min, max, avg, std for all sensor columns for a specific device.

    TIME RANGE RULES:
    - "status" / "overview" / "summary" / "condition" / "listen" / "how is"
      -> all_time=False, hours=24  (last 24 hours only)
    - "max" / "min" / "average" / "avg" / "std" / "highest" / "lowest" / "ever" / "all time"
      -> all_time=True  (full dataset, no time filter)

    Args:
        device_id: Normalized device ID, dashes removed. e.g. 'IEMA60120012'.
        hours: Hours to look back. Used only when all_time=False. Default 24.
        all_time: If True, queries entire table with no time filter. Default False.
    """
    engine = get_engine()
    label = "all time (full dataset)" if all_time else f"last {hours} hours"
    try:
        df = pd.read_sql(_sensor_query(device_id, all_time, hours), engine)
    except Exception as e:
        return f"Error fetching sensor data for '{device_id}': {e}"

    if df.empty or df.isnull().all(axis=None):
        return f"No sensor data found for device '{device_id}' ({label})."

    return _format_block(df.iloc[0], device_id, _fetch_meta(device_id), label)


@tool
def get_all_devices_status(hours: int = 24) -> str:
    """
    Get sensor stats (min, max, avg, std) for ALL devices in the system,
    based on the last N hours of data (default: last 24 hours).

    Use this tool when the user asks about ALL devices with no specific filter:
    - "status of all devices"
    - "overview of all devices"
    - "give me status of every device"
    - "show all device status"
    - "summarize all devices"
    - "how are all devices doing"
    - "overview of everything"
    - "all devices performance"

    Args:
        hours: How many hours back to query. Default is 24 (last 24 hours).
    """
    label = f"last {hours} hours"
    engine = get_engine()

    df_dev = _fetch_all_devices()
    if df_dev.empty:
        return "No devices found in the system."

    results = [f"Status of ALL devices ({label}) — {len(df_dev)} device(s) found:\n"]
    results.append("=" * 60)

    for _, r in df_dev.iterrows():
        did = r["device_id"]
        asset_name = r["asset_name"]

        # Skip devices whose ID would produce an invalid table name
        if not did or did.strip() == "":
            results.append(f"\nDevice '{asset_name}': Skipped — invalid device ID.")
            results.append("-" * 60)
            continue

        try:
            df_s = pd.read_sql(_sensor_query(did, all_time=False, hours=hours), engine)
        except Exception as e:
            err_short = str(e).split("\n")[0]  # first line only, no giant SQL dump
            results.append(f"\nDevice {did} ({asset_name}): No sensor table found — skipped.")
            results.append("-" * 60)
            continue

        if df_s.empty or df_s.isnull().all(axis=None):
            results.append(f"\nDevice {did} ({asset_name}): No data in the last {hours} hours.")
            results.append("-" * 60)
            continue

        meta_line = (
            f"Asset: {asset_name} | Site: {r['site']} | "
            f"Application: {r['application']} | "
            f"Workshop: {r['workshop_name']} ({r['workshop_address']}) | "
            f"Company: {r['company_name']} ({r['company_address']})"
        )
        results.append("\n" + _format_block(df_s.iloc[0], did, meta_line, label))
        results.append("-" * 60)

    return "\n".join(results)


@tool
def resolve_and_get_status(filter_value: str, hours: int = 24, all_time: bool = False) -> str:
    """
    Resolve a company/site/application/workshop filter to device IDs, then
    return sensor stats for ALL matching devices.

    Searches across: company_name, company_address, workshop_name,
    workshop_address, site, application, asset_type.

    TIME RANGE RULES:
    - "status" / "overview" / "summary" / "condition" / "listen" / "how is"
      -> all_time=False, hours=24  (last 24 hours only)
    - "max" / "min" / "average" / "avg" / "std" / "highest" / "lowest" / "ever" / "all time"
      -> all_time=True  (full dataset, no time filter)

    Args:
        filter_value: e.g. 'Kolkata', 'DRI', 'DUO_', 'Jai Balaji Group'.
        hours: Hours to look back. Used only when all_time=False. Default 24.
        all_time: If True, queries entire table with no time filter. Default False.
    """
    label = "all time (full dataset)" if all_time else f"last {hours} hours"
    engine = get_engine()

    df_dev = _resolve_devices(filter_value)
    if df_dev.empty:
        return f"No devices found matching: '{filter_value}'."

    results = []
    for _, r in df_dev.iterrows():
        did = r["device_id"]
        asset_name = r["asset_name"]

        if not did or did.strip() == "":
            results.append(f"Device '{asset_name}': Skipped — invalid device ID.")
            continue

        try:
            df_s = pd.read_sql(_sensor_query(did, all_time, hours), engine)
        except Exception:
            results.append(f"Device {did} ({asset_name}): No sensor table found — skipped.")
            continue

        if df_s.empty or df_s.isnull().all(axis=None):
            results.append(f"Device {did} ({asset_name}): No data ({label}).")
            continue

        meta_line = (
            f"Asset: {asset_name} | Site: {r['site']} | "
            f"Application: {r['application']} | "
            f"Workshop: {r['workshop_name']} ({r['workshop_address']}) | "
            f"Company: {r['company_name']} ({r['company_address']})"
        )
        results.append(_format_block(df_s.iloc[0], did, meta_line, label))

    return "\n\n".join(results)


@tool
def list_devices(filter_value: str) -> str:
    """
    List all devices/assets matching a filter across company, workshop, and asset tables.
    Use for: 'devices in DRI', 'assets in Kolkata', 'device present in X',
    'list assets in Jai Balaji Group', 'show all devices in company Y'.

    Searches: company_name, company_address, workshop_name, workshop_address,
    site, application, asset_type.

    Args:
        filter_value: e.g. 'DRI', 'Kolkata', 'DUO_', 'Jai Balaji Group'.
    """
    df = _resolve_devices(filter_value)

    if df.empty:
        return f"No devices found matching: '{filter_value}'."

    lines = [f"Found {len(df)} device(s) matching '{filter_value}':\n"]
    for _, r in df.iterrows():
        lines.append(
            f"  Asset: {r['asset_name']} | Device ID: {r['device_id']} | "
            f"Type: {r['asset_type']} | Site: {r['site']} | "
            f"Application: {r['application']} | "
            f"Workshop: {r['workshop_name']} ({r['workshop_address']}) | "
            f"Company: {r['company_name']} ({r['company_address']})"
        )
    return "\n".join(lines)


@tool
def list_all_devices() -> str:
    """
    List every device/asset registered in the system, with full metadata.
    Use when the user asks to see all devices with no filter:
    - "list all devices"
    - "show all assets"
    - "what devices are in the system"
    - "how many devices are there"
    """
    df = _fetch_all_devices()

    if df.empty:
        return "No devices found in the system."

    lines = [f"Total devices in system: {len(df)}\n"]
    for _, r in df.iterrows():
        lines.append(
            f"  Asset: {r['asset_name']} | Device ID: {r['device_id']} | "
            f"Type: {r['asset_type']} | Site: {r['site']} | "
            f"Application: {r['application']} | "
            f"Workshop: {r['workshop_name']} ({r['workshop_address']}) | "
            f"Company: {r['company_name']} ({r['company_address']})"
        )
    return "\n".join(lines)


@tool
def get_company_info(filter_value: str) -> str:
    """
    Get company details along with their workshops and devices.
    Use for ANY query about a company — company name, address, contact,
    workshops under a company, devices under a company.
    Examples: 'info about Jai Balaji Group', 'company in Kolkata',
    'which company owns workshop DUO_', 'details of company XYZ'.

    Args:
        filter_value: Company name, address, or any company column value.
                      e.g. 'Jai Balaji Group', 'Kolkata'.
    """
    engine = get_engine()

    company_cols = ["company_name", "company_address", "company_description", "company_email", "company_mobile"]

    conditions, params = [], []
    for col in company_cols:
        conditions.append(f"LOWER(c.{col}) = LOWER(%s)")
        params.append(filter_value)
    # Also allow partial match on company_name for usability
    conditions.append(f"LOWER(c.company_name) LIKE LOWER(%s)")
    params.append(f"%{filter_value}%")

    q = f"""
        SELECT DISTINCT
               c.company_id, c.company_name, c.company_address,
               c.company_description, c.company_email, c.company_mobile,
               w.workshop_id, w.workshop_name, w.workshop_address,
               a.asset_name, REPLACE(a.asset_name, '-', '') AS device_id,
               a.asset_type, a.site, a.application
        {_BASE_JOIN}
        WHERE {" OR ".join(conditions)}
        ORDER BY c.company_name, w.workshop_name, a.asset_name
    """
    try:
        df = pd.read_sql(q, engine, params=tuple(params))
    except Exception as e:
        return f"Error fetching company info for '{filter_value}': {e}"

    if df.empty:
        return f"No company found matching: '{filter_value}'."

    lines = []
    for company_id, cgroup in df.groupby("company_id"):
        cr = cgroup.iloc[0]
        lines.append(
            f"Company: {cr['company_name']}\n"
            f"  Address:     {cr.get('company_address', 'N/A')}\n"
            f"  Description: {cr.get('company_description', 'N/A')}\n"
            f"  Email:       {cr.get('company_email', 'N/A')}\n"
            f"  Mobile:      {cr.get('company_mobile', 'N/A')}\n"
        )
        for wid, wgroup in cgroup.groupby("workshop_id"):
            wr = wgroup.iloc[0]
            devices = [
                f"{r['asset_name']} ({r['device_id']})"
                for _, r in wgroup.iterrows()
                if pd.notna(r.get("asset_name"))
            ]
            lines.append(
                f"  Workshop: {wr['workshop_name']} | Address: {wr['workshop_address']}\n"
                f"    Devices ({len(devices)}): {', '.join(devices) if devices else 'None'}"
            )
        lines.append("")  # blank line between companies

    return "\n".join(lines).strip()


@tool
def get_workshop_details(filter_value: str) -> str:
    """
    Get workshop details along with company info and associated devices.
    Use for: 'workshop in Kolkata', 'details of DUO_ workshop', 'list workshops'.

    Args:
        filter_value: Workshop name, address, or city e.g. 'Kolkata', 'DUO_'.
    """
    engine = get_engine()

    q = f"""
        SELECT w.workshop_id, w.workshop_name, w.workshop_address,
               w.workshop_description,
               c.company_name, c.company_address,
               a.asset_name, a.asset_type, a.site, a.application
        {_BASE_JOIN}
        WHERE LOWER(w.workshop_name)        = LOWER(%s)
           OR LOWER(w.workshop_address)     = LOWER(%s)
           OR LOWER(w.workshop_description) = LOWER(%s)
        ORDER BY w.workshop_name, a.asset_name
    """
    try:
        df = pd.read_sql(q, engine, params=(filter_value, filter_value, filter_value))
    except Exception as e:
        return f"Error fetching workshop details for '{filter_value}': {e}"

    if df.empty:
        return f"No workshops found matching: '{filter_value}'."

    lines = []
    for wid, group in df.groupby("workshop_id"):
        wr = group.iloc[0]
        devices = [r["asset_name"] for _, r in group.iterrows() if pd.notna(r["asset_name"])]
        lines.append(
            f"Workshop: {wr['workshop_name']} | Address: {wr['workshop_address']} | "
            f"Description: {wr['workshop_description']}\n"
            f"  Company: {wr['company_name']} ({wr['company_address']})\n"
            f"  Devices ({len(devices)}): {', '.join(devices) if devices else 'None'}"
        )
    return "\n\n".join(lines)