













import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from datetime import datetime, timedelta
from config import engine, OUTPUT_DIR
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json

# --- Load API Key ---
load_dotenv(override=True)
API_KEY = os.getenv("API_KEY")

# --- Parameter Extraction using LLM ---
def extract_parameters(user_input: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, api_key=API_KEY
    )
    current_time = datetime.now()
    current_date_str = current_time.strftime("%Y-%m-%d %H:%M")
    
    prompt = ChatPromptTemplate.from_template(
        """Current time is {current_time}. Current year is 2025.
        
Extract from the user query for anomaly detection or comparison:
- Device name (e.g., 'IEMA-601-2-0012', normalize by removing dashes to 'IEMA60120012').
- If no device name, extract filter values (e.g., 'Kolkata', 'Jai Balaji Group', 'DRI', 'ESP ID FAN 2', 'a').
- Column to plot (e.g., 'temperature_one', 'temperature_two', 'vibration_x', 'vibration_y', 'vibration_z') if specified.
- Time range: start and end times in 'YYYY-MM-DD HH:MM' format.
  - If relative (e.g., 'last 1 day', 'last 24 hours'), compute start as current minus 24 hours, end as current.
  - If 'between A and B', parse A and B.
  - If end is only time (e.g., '23:00'), prepend date from start or current.
  - If start is 'from HH:MM' on a date (e.g., '27th Aug from 3pm'), set start to that, end to same day 23:59.
  - If only date is provided (e.g., '27th Aug'), set start to 00:00, end to 23:59 on that day.
  - If no time specified, default to last 24 hours.
  - If date is provided without year (e.g., '27th Aug'), assume current year (2025).

Output strict JSON (no extra text, no backticks): 
{{
  "device": "normalized_id or null",
  "filter_values": ["value1", "value2", ...] or null,
  "column": "column_name or null",
  "start_time": "YYYY-MM-DD HH:MM",
  "end_time": "YYYY-MM-DD HH:MM"
}}

Query: {query}"""
    )
    
    chain = prompt | llm
    response = chain.invoke({"current_time": current_date_str, "query": user_input})
    
    try:
        json_str = re.sub(r'^```json\n|\n```$', '', response.content.strip())
        params = json.loads(json_str)
        
        # Validate time range
        if not params.get("start_time"):
            params["start_time"] = (current_time - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")
        if not params.get("end_time"):
            params["end_time"] = current_date_str
        
        # Handle partial time
        if params["start_time"] and not params.get("end_time"):
            start_dt = datetime.strptime(params["start_time"], "%Y-%m-%d %H:%M")
            params["end_time"] = start_dt.strftime("%Y-%m-%d") + " 23:59"
        elif params["end_time"] and not params.get("start_time"):
            end_dt = datetime.strptime(params["end_time"], "%Y-%m-%d %H:%M")
            params["start_time"] = end_dt.strftime("%Y-%m-%d") + " 00:00"
        
        # Handle relative time
        if "last" in user_input.lower():
            match = re.search(r"last\s+(24\s+hours|\d+\s+days?)", user_input.lower())
            if match:
                params["end_time"] = current_date_str
                params["start_time"] = (current_time - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")
        
        # Handle date like "27th Aug from 3pm"
        date_match = re.search(r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\s*(?:from\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?", user_input, re.IGNORECASE)
        if date_match:
            date_str = date_match.group(1)
            time_str = date_match.group(2) if date_match.group(2) else "00:00"
            if time_str:
                time_match = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_str, re.IGNORECASE)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2) or "00")
                    period = time_match.group(3).lower() if time_match.group(3) else None
                    if period == "pm" and hour != 12:
                        hour += 12
                    elif period == "am" and hour == 12:
                        hour = 0
                    time_str = f"{hour:02d}:{minute:02d}"
            try:
                date_dt = datetime.strptime(f"{date_str} 2025", "%d %b %Y")
                params["start_time"] = date_dt.strftime("%Y-%m-%d") + f" {time_str}"
                params["end_time"] = date_dt.strftime("%Y-%m-%d") + " 23:59"
            except ValueError:
                pass
        
        return params
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {
            "device": None,
            "filter_values": None,
            "column": None,
            "start_time": (current_time - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"),
            "end_time": current_date_str
        }

# --- Resolve Device IDs using SQL ---
def resolve_devices(filter_values: List[str]) -> list:
    if not filter_values:
        return []

    asset_columns = ["asset_name", "site", "application", "asset_type", "class", "asset_description"]
    query_assets = """
        SELECT a.asset_id, a.asset_name, 
               REPLACE(a.asset_name, '-', '') AS device_id,
               a.asset_type, a.site, a.application, 
               a.asset_description, a.class,
               w.workshop_id, w.workshop_name, w.workshop_address, w.workshop_description, w.workshop_image
        FROM assets a
        LEFT JOIN workshops w ON a.workshop_id_fk = w.workshop_id
        WHERE {}
    """
    conditions = []
    params = []
    for fv in filter_values:
        for col in asset_columns:
            conditions.append(f"LOWER(a.{col}) = LOWER(%s)")
            params.append(fv.lower())
    query_assets = query_assets.format(" OR ".join(conditions))

    try:
        df_assets = pd.read_sql(query_assets, engine, params=tuple(params))
    except Exception as e:
        print(f"Error searching assets: {e}")
        df_assets = pd.DataFrame()

    workshop_columns = ["workshop_name", "workshop_address", "workshop_description"]
    query_workshops = """
        SELECT a.asset_id, a.asset_name, 
               REPLACE(a.asset_name, '-', '') AS device_id,
               a.asset_type, a.site, a.application, 
               a.asset_description, a.class,
               w.workshop_id, w.workshop_name, w.workshop_address, w.workshop_description, w.workshop_image
        FROM assets a
        LEFT JOIN workshops w ON a.workshop_id_fk = w.workshop_id
        WHERE {}
    """
    conditions = []
    params = []
    for fv in filter_values:
        for col in workshop_columns:
            conditions.append(f"LOWER(w.{col}) = LOWER(%s)")
            params.append(fv.lower())
    query_workshops = query_workshops.format(" OR ".join(conditions))

    try:
        df_workshops = pd.read_sql(query_workshops, engine, params=tuple(params))
    except Exception as e:
        print(f"Error searching workshops: {e}")
        df_workshops = pd.DataFrame()

    df = pd.concat([df_assets, df_workshops]).drop_duplicates(subset=['device_id'])
    return df.to_dict(orient="records") if not df.empty else []

# --- Fetch Metadata ---
def get_asset_workshop_metadata(device: str) -> dict:
    query = """
        SELECT a.asset_id, a.asset_name, a.asset_type, a.site, a.application, 
               a.asset_description, a.class, w.workshop_id, w.workshop_name, 
               w.workshop_address, w.workshop_description, w.workshop_image
        FROM assets a
        LEFT JOIN workshops w ON a.workshop_id_fk = w.workshop_id
        WHERE REPLACE(a.asset_name, '-', '') = %s
    """
    df = pd.read_sql(query, engine, params=(device,))
    if df.empty:
        return {k: "Unknown" for k in ["asset_id", "asset_name", "asset_type", "site", "application", "asset_description", "class", "workshop_name", "workshop_address", "workshop_description", "workshop_image"]}
    return df.iloc[0].to_dict()

# --- Fetch Anomalies ---
def get_anomalies(device: str, start_time: str, end_time: str) -> pd.DataFrame:
    table = f"anomaly_data_{device}"
    query = f"""
        SELECT starttimestamp, endtimestamp, temperature_status, vibration_status, overall_health
        FROM {table}
        WHERE (temperature_status='unhealthy' OR vibration_status='unhealthy' OR overall_health='Unhealthy')
          AND starttimestamp <= '{end_time}' AND endtimestamp >= '{start_time}'
        ORDER BY starttimestamp ASC;
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error fetching anomalies for {device}: {e}")
        return pd.DataFrame()

# --- Fetch Sensor Data ---
def get_sensor_data(device: str, start_time: str, end_time: str) -> pd.DataFrame:
    table = f"sensor_data_{device}"
    query = f"""
        SELECT timestamp, temperature_one, temperature_two, vibration_x, vibration_y, vibration_z
        FROM {table}
        WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY timestamp ASC;
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error fetching sensor data for {device}: {e}")
        return pd.DataFrame()

# --- Decide Columns to Plot ---
def get_columns_to_plot(anomalies: pd.DataFrame, check_anomalies: bool, user_column: str = None) -> list:
    if user_column and user_column in ["temperature_one", "temperature_two", "vibration_x", "vibration_y", "vibration_z"]:
        return [user_column]
    if check_anomalies and not anomalies.empty:
        cols = set()
        if "temperature_status" in anomalies.columns and (anomalies["temperature_status"] == "unhealthy").any():
            cols.update(["temperature_one", "temperature_two"])
        if "vibration_status" in anomalies.columns and (anomalies["vibration_status"] == "unhealthy").any():
            cols.update(["vibration_x", "vibration_y", "vibration_z"])
        return list(cols)
    return ["temperature_one", "temperature_two", "vibration_x", "vibration_y", "vibration_z"]

# --- Plot with Anomaly Shading ---
def plot_with_anomalies(df: pd.DataFrame, device: str, anomalies: pd.DataFrame, metadata: dict, check_anomalies: bool, user_column: str = None) -> Tuple[str, List[str]]:
    if df.empty:
        info = f" ({metadata['asset_name']}, Site: {metadata['site']}, Application: {metadata['application']})"
        workshop = f" in Workshop {metadata.get('workshop_id', 'Unknown')} ({metadata['workshop_name']}, {metadata['workshop_address']})"
        return f"No sensor data available for {device}{info}{workshop}.", []

    cols_to_plot = get_columns_to_plot(anomalies, check_anomalies, user_column)
    if not cols_to_plot:
        return f"No relevant columns to plot for {device}.", []

    graph_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if check_anomalies and not anomalies.empty:
        anomalies['starttimestamp'] = pd.to_datetime(anomalies['starttimestamp'])
        anomalies['endtimestamp'] = pd.to_datetime(anomalies['endtimestamp'])

    # Temperature
    temp_cols = [c for c in cols_to_plot if c in ["temperature_one", "temperature_two"]]
    if temp_cols:
        plt.figure(figsize=(10, 4))
        for i, col in enumerate(temp_cols):
            if col in df.columns:
                plt.plot(df["timestamp"], df[col], label=col, color=["blue", "green"][i % 2])
        if check_anomalies and not anomalies.empty:
            for i, row in anomalies.iterrows():
                plt.axvspan(row["starttimestamp"], row["endtimestamp"], color="#ff4d4d", alpha=0.45, zorder=1+i)
        plt.title(f"{device} - Temperature")
        plt.xlabel("Timestamp")
        plt.ylabel("Temperature")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"){device}_temperature_{ts}.png")
        plt.savefig(fname)
        plt.close()
        graph_paths.append(fname)

    # Vibration
    vib_cols = [c for c in cols_to_plot if c in ["vibration_x", "vibration_y", "vibration_z"]]
    if vib_cols:
        plt.figure(figsize=(10, 4))
        colors = ["blue", "green", "orange"]
        for i, col in enumerate(vib_cols):
            if col in df.columns:
                plt.plot(df["timestamp"], df[col], label=col, color=colors[i % len(colors)])
        if check_anomalies and not anomalies.empty:
            for i, row in anomalies.iterrows():
                plt.axvspan(row["starttimestamp"], row["endtimestamp"], color="#ff4d4d", alpha=0.45, zorder=1+i)
        plt.title(f"{device} - Vibration")
        plt.xlabel("Timestamp")
        plt.ylabel("Vibration")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"{device}_vibration_{ts}.png")
        plt.savefig(fname)
        plt.close()
        graph_paths.append(fname)

    info = f" ({metadata['asset_name']}, Site: {metadata['site']}, Application: {metadata['application']})"
    workshop = f" in Workshop {metadata.get('workshop_id', 'Unknown')} ({metadata['workshop_name']}, {metadata['workshop_address']})"
    return f"Graphs generated for sensor data of {device}{info}{workshop}", graph_paths

# --- Comparison Plot ---
def plot_comparison(devices: List[dict], column: str, start_time: str, end_time: str) -> Tuple[str, List[str]]:
    if not devices or not column:
        return "No devices or column for comparison.", []
    valid = ["temperature_one", "temperature_two", "vibration_x", "vibration_y", "vibration_z"]
    if column not in valid:
        return f"Invalid column '{column}'. Choose from {', '.join(valid)}.", []

    plt.figure(figsize=(12, 6))
    colors = ["blue", "green", "orange", "red", "purple"]
    summaries = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, dev in enumerate(devices):
        did = dev["device_id"]
        meta = get_asset_workshop_metadata(did)
        df = get_sensor_data(did, start_time, end_time)
        if df.empty or column not in df.columns:
            summaries.append(f"No {column} data for {did}.")
            continue
        plt.plot(df["timestamp"], df[column], label=f"{did} ({meta['asset_name']})", color=colors[i % len(colors)])
        summaries.append(f"Plotted {column} for {did}")

    if not plt.get_fignums():
        plt.close()
        return "\n".join(summaries) if summaries else "No data for comparison.", []

    plt.title(f"Comparison: {column.replace('_', ' ').title()}")
    plt.xlabel("Timestamp")
    plt.ylabel(column.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"comparison_{column}_{ts}.png")
    plt.savefig(fname)
    plt.close()
    return "\n".join(summaries), [fname]

# --- Main Runner ---
def run_anomaly_detection(user_input: str) -> tuple:
    try:
        params = extract_parameters(user_input)
        device_id = params.get("device")
        filter_values = params.get("filter_values")
        column = params.get("column")
        start_time = params.get("start_time")
        end_time = params.get("end_time")

        # ALWAYS show anomalies in red
        check_anomalies = True

        is_comparison = "comparison" in user_input.lower() or (filter_values and len(filter_values) > 1)

        if device_id:
            devices = [{"device_id": device_id}]
        else:
            devices = resolve_devices(filter_values or [])
            if not devices:
                return f"No devices found for: '{', '.join(filter_values or [])}'.", []

        if is_comparison and column:
            return plot_comparison(devices, column, start_time, end_time)

        summaries = []
        all_graph_paths = []
        for dev in devices:
            did = dev["device_id"]
            meta = get_asset_workshop_metadata(did)

            # Always fetch anomalies
            anomalies = get_anomalies(did, start_time, end_time)

            # Warn only if user asked for anomalies and none found
            if "anomalies" in user_input.lower() and anomalies.empty:
                info = f" ({meta['asset_name']}, Site: {meta['site']}, Application: {meta['application']})"
                workshop = f" in Workshop {meta.get('workshop_id', 'Unknown')} ({meta['workshop_name']})"
                summaries.append(f"No anomalies found for {did}{info}{workshop}.")
                continue

            df = get_sensor_data(did, start_time, end_time)
            if df.empty:
                info = f" ({meta['asset_name']}, Site: {meta['site']}, Application: {meta['application']})"
                workshop = f" in Workshop {meta.get('workshop_id', 'Unknown')} ({meta['workshop_name']})"
                summaries.append(f"No sensor data for {did}{info}{workshop}.")
                continue

            summary, paths = plot_with_anomalies(df, did, anomalies, meta, check_anomalies, column)
            summaries.append(summary)
            all_graph_paths.extend(paths)

        return ("\n".join(summaries), all_graph_paths) if summaries else ("No results.", [])

    except Exception as e:
        return f"Error: {e}", []