





import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# --- Load Environment Variables ---
load_dotenv(override=True)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_DRIVER = os.getenv("DB_DRIVER", "mysqlconnector")

# --- Database Connection ---
db_uri = f"mysql+{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_uri)

# --- Output Directory ---
OUTPUT_DIR = "device_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Metadata Resolver ---
def resolve_device_ids(filters: dict) -> list:
    """
    Resolve user filters into a list of device_ids with metadata.
    Now: if user provides any location/company name, we check both
    workshop_address (in workshops) and application (in assets).
    """
    # Split normal filters
    asset_filters = {k: v for k, v in filters.items() if k in ["asset_name", "site", "asset_type", "class"]}
    
    # Treat ANY free-text location/company name as both workshop_address and application
    location_filters = {k: v for k, v in filters.items() if k not in asset_filters}

    query = """
        SELECT a.asset_name, 
               REPLACE(a.asset_name, '-', '') AS device_id,
               a.asset_type, a.site, a.application, 
               a.asset_description, a.class,
               w.workshop_id, w.workshop_address
        FROM assets a
        LEFT JOIN workshops w ON a.workshop_id_fk = w.workshop_id
        WHERE 1=1
    """

    params = []
    if location_filters:
        for _, val in location_filters.items():
            # Search in both workshop_address and application
            query += " AND (w.workshop_address LIKE %s OR a.application LIKE %s)"
            params.append(f"%{val}%")
            params.append(f"%{val}%")

    if asset_filters:
        for col, val in asset_filters.items():
            query += f" AND a.{col} LIKE %s"
            params.append(f"%{val}%")

    df = pd.read_sql(query, engine, params=params)

    if df.empty:
        return []

    return df.to_dict(orient="records")
