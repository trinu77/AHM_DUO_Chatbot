

# # app.py
# import streamlit as st
# import time
# from datetime import datetime
# import os

# # ─── Import your existing modules ───────────────────────────────────────
# from a import create_agent
# from b import run_anomaly_detection
# from config import OUTPUT_DIR

# # ─── Page config ─────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Device Monitor & Anomaly Bot",
#     page_icon="🔧",
#     layout="wide"
# )

# st.title("🔧 Device Data & Anomaly Detection Assistant")
# st.caption(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Location aware")

# # ─── Initialize session state ────────────────────────────────────────────
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "graph" not in st.session_state:
#     with st.spinner("Loading agent... (first run may take 10–30 seconds)"):
#         st.session_state.graph = create_agent()

# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = f"thread_{int(time.time())}"

# # ─── Helper to extract readable text from possibly structured content ────
# def extract_response_text(msg):
#     """Extract clean text from various possible message formats"""
#     if hasattr(msg, 'content'):
#         content = msg.content
#     else:
#         content = msg

#     # Case 1: already a clean string
#     if isinstance(content, str):
#         return content.strip()

#     # Case 2: list of dicts → most common with recent Gemini / Google models
#     if isinstance(content, list):
#         texts = []
#         for item in content:
#             if isinstance(item, dict):
#                 # Look for 'text' field (common in structured responses)
#                 if 'text' in item:
#                     texts.append(item['text'])
#                 elif item.get('type') == 'text':
#                     texts.append(item.get('text', ''))
#             elif isinstance(item, str):
#                 texts.append(item)
#         return "\n\n".join(filter(None, texts)).strip()

#     # Case 3: dict with 'text' key
#     if isinstance(content, dict):
#         return content.get('text', str(content)).strip()

#     # Fallback
#     return str(content).strip()


# # ─── Display chat history ────────────────────────────────────────────────
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if message.get("images"):
#             cols = st.columns(min(3, len(message["images"])))
#             for idx, img_path in enumerate(message["images"]):
#                 with cols[idx]:
#                     st.image(
#                         img_path,
#                         use_column_width=True,
#                         caption=os.path.basename(img_path)
#                     )

# # ─── Chat input ───────────────────────────────────────────────────────────
# if prompt := st.chat_input("Ask about device status, anomalies, plots, assets, workshops…"):

#     # Add user message to history
#     st.session_state.messages.append({
#         "role": "user",
#         "content": prompt,
#         "images": []
#     })

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # ─── Decide which backend to use ─────────────────────────────────────
#     is_plot_request = any(
#         kw in prompt.lower() for kw in [
#             "plot", "graph", "chart", "anomaly", "anomalies",
#             "visualize", "show trend", "draw", "figure", "image"
#         ]
#     )

#     with st.chat_message("assistant"):
#         placeholder = st.empty()
#         response_text = ""
#         response_images = []

#         if is_plot_request:
#             with st.spinner("Analyzing anomalies & generating plots..."):
#                 try:
#                     summary, graph_paths = run_anomaly_detection(prompt)

#                     # Ensure summary is string
#                     summary = str(summary).strip() if summary else "No summary available."

#                     response_text = summary
#                     if graph_paths:
#                         response_text += "\n\n**Generated graphs:**"

#                     response_images = graph_paths if graph_paths else []

#                 except Exception as e:
#                     response_text = f"⚠️ Error during plotting/anomaly detection:\n{str(e)}"

#         else:
#             # Text-only agent path (LangGraph)
#             with st.spinner("Thinking..."):
#                 try:
#                     config = {
#                         "configurable": {
#                             "thread_id": st.session_state.thread_id,
#                             "recursion_limit": 60
#                         }
#                     }

#                     result = st.session_state.graph.invoke(
#                         {"messages": [{"role": "user", "content": prompt}]},
#                         config=config
#                     )

#                     # Extract clean text from possibly complex message structure
#                     response_text = extract_response_text(result["messages"][-1])

#                 except Exception as e:
#                     response_text = f"⚠️ Agent error: {str(e)}"

#         # ─── Display final response ──────────────────────────────────────
#         placeholder.markdown(response_text)

#         # Show images if any
#         if response_images:
#             cols = st.columns(min(3, len(response_images)))
#             for i, path in enumerate(response_images):
#                 with cols[i % len(cols)]:
#                     st.image(path, use_column_width=True, caption=os.path.basename(path))

#     # ─── Save assistant response to history ──────────────────────────────
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": response_text,
#         "images": response_images
#     })

#     # Optional: force scroll to bottom
#     # st.rerun()







# app.py
import streamlit as st
import time
from datetime import datetime
import os

# ─── Import your existing modules
from a import create_agent
from data import run_anomaly_detection
from config import OUTPUT_DIR

# ─── Page config 
st.set_page_config(
    page_title="Device Monitor & Anomaly Bot",
    page_icon="🔧",
    layout="wide"
)

st.title(" Device Data & Anomaly Detection Assistant")
st.caption(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Location aware")

# ─── Initialize session state 
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread_{int(time.time())}"

# ─── Add debug prints around agent creation ──────────────────────────────
if "graph" not in st.session_state:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating LangGraph agent for the FIRST time...")
    import time  # Already imported at top, but safe
    start = time.time()
    with st.spinner("Loading agent... (first run may take 10–30 seconds)"):
        st.session_state.graph = create_agent()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Agent created in {time.time() - start:.1f} seconds")
else:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] → Re-using existing agent from session_state")

# ─── Helper to extract readable text from possibly structured content ────
def extract_response_text(msg):
    """Extract clean text from various possible message formats"""
    if hasattr(msg, 'content'):
        content = msg.content
    else:
        content = msg

    # Case 1: already a clean string
    if isinstance(content, str):
        return content.strip()

    # Case 2: list of dicts → most common with recent Gemini / Google models
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                # Look for 'text' field (common in structured responses)
                if 'text' in item:
                    texts.append(item['text'])
                elif item.get('type') == 'text':
                    texts.append(item.get('text', ''))
            elif isinstance(item, str):
                texts.append(item)
        return "\n\n".join(filter(None, texts)).strip()

    # Case 3: dict with 'text' key
    if isinstance(content, dict):
        return content.get('text', str(content)).strip()

    # Fallback
    return str(content).strip()

# ─── Display chat history ────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("images"):
            cols = st.columns(min(3, len(message["images"])))
            for idx, img_path in enumerate(message["images"]):
                with cols[idx]:
                    st.image(
                        img_path,
                        use_column_width=True,
                        caption=os.path.basename(img_path)
                    )

# ─── Chat input ───────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about device status, anomalies, plots, assets, workshops…"):

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "images": []
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # ─── Decide which backend to use ─────────────────────────────────────
    is_plot_request = any(
        kw in prompt.lower() for kw in [
            "plot", "graph", "chart", "anomaly", "anomalies",
            "visualize", "show trend", "draw", "figure", "image"
        ]
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""
        response_images = []

        if is_plot_request:
            with st.spinner("Analyzing anomalies & generating plots..."):
                try:
                    summary, graph_paths = run_anomaly_detection(prompt)

                    # Ensure summary is string
                    summary = str(summary).strip() if summary else "No summary available."

                    response_text = summary
                    if graph_paths:
                        response_text += "\n\n**Generated graphs:**"

                    response_images = graph_paths if graph_paths else []

                except Exception as e:
                    response_text = f"⚠️ Error during plotting/anomaly detection:\n{str(e)}"

        else:
            # Text-only agent path (LangGraph)
            with st.spinner("Thinking..."):
                try:
                    config = {
                        "configurable": {
                            "thread_id": st.session_state.thread_id,
                            "recursion_limit": 60
                        }
                    }

                    result = st.session_state.graph.invoke(
                        {"messages": [{"role": "user", "content": prompt}]},
                        config=config
                    )

                    # Extract clean text from possibly complex message structure
                    response_text = extract_response_text(result["messages"][-1])

                except Exception as e:
                    response_text = f"⚠️ Agent error: {str(e)}"

        # ─── Display final response ──────────────────────────────────────
        placeholder.markdown(response_text)

        # Show images if any
        if response_images:
            cols = st.columns(min(3, len(response_images)))
            for i, path in enumerate(response_images):
                with cols[i % len(cols)]:
                    st.image(path, use_column_width=True, caption=os.path.basename(path))

    # ─── Save assistant response to history ──────────────────────────────
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "images": response_images
    })

    # Optional: force scroll to bottom
    # st.rerun()