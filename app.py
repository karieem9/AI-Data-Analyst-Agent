import pandas as pd
import streamlit as st

from profiling import profile_dataframe

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
st.title("AI Data Analyst Agent")

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Couldn't read that file: {e}")
        st.stop()

    st.session_state["df"] = df

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Profile")
    st.dataframe(profile_dataframe(df), use_container_width=True)
else:
    st.info("Upload a file to see its profile.")
