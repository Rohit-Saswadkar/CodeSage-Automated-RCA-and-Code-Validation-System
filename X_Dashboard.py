print('Xperimental_platform:-')
print('Improved llms reasoning for self identified exception')

from dotenv import load_dotenv
from langchain.chains import LLMChain
load_dotenv()
import os
import time
import datetime
import subprocess
import pandas as pd
from datetime import date



dt_now = datetime.datetime.now()




import streamlit as st
st.set_page_config(layout="wide")


# print('V1: W/O feedback')
#
# elif st.session_state.level == "rca_view":
#     st.subheader("🧾 Detailed RCA and Solution")
#
#     content = st.session_state.selected_rca
#
#
#     with st.container():
#         st.markdown("---")
#         st.markdown("#### 📄 RCA and Solution Report")
#         st.markdown(content, unsafe_allow_html=False)
#         st.markdown("---")
#
#     st.button("🔙 Back to Details", on_click=lambda: st.session_state.update(level="details"))

# print('V2: With Feedback')








st.title("AI Debugging Dashboard")

latest_file_name = None

data_path = os.getenv("data_path_2")
print('Data path', data_path)
input_folder = 'Test_Results/6_RCA/'
output_folder = 'Test_Results/7_Feedback/'




def get_latest_csv_filename(folder_path: str, pattern: str = "_AI_debugging.csv"):
    files = [f for f in os.listdir(folder_path) if f.endswith(pattern)]
    if not files:
        return None
    files = [os.path.join(folder_path, f) for f in files]
    return max(files, key=os.path.getmtime)



# Date Inputs
start_date = st.date_input("Start Date", value=date.today())
end_date = st.date_input("End Date", value=date.today())

col1, col2 = st.columns(2)

if col1.button("🔄 Run Pipeline and Load Logs"):
    latest_file_name = f"{dt_now}_AI_debugging.csv"
    print("Latest file name:-", latest_file_name)

    if start_date > end_date:
        st.error("Start date must be earlier than or equal to end date.")
    else:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_file_name = f"{timestamp_str}_AI_debugging.csv"
        full_output_path = os.path.join(data_path, input_folder, latest_file_name)

        with st.spinner("Running log pipeline..."):
            result = subprocess.run([
                "python3",
                "/home/rohitsaswadkar/PycharmProjects/PythonProject/AI/Projects/R_3.0/Agentic_Error_solver/Local_system/local_pipeline/X_pipeline.py",
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                latest_file_name
            ], capture_output=True, text=True)

            st.text_area("Pipeline Output", result.stdout + "\n" + result.stderr, height=300)

        max_wait = 60
        waited = 0
        while not os.path.exists(full_output_path) and waited < max_wait:
            time.sleep(2)
            waited += 2

        if not os.path.exists(full_output_path):
            st.error("❌ Output file was not created.")
            st.stop()

        df = pd.read_csv(full_output_path)
        st.session_state.df = df
        st.success(f"✅ Loaded file: {latest_file_name}")

# --- Skip Pipeline and Load Latest File ---
elif col2.button("⏭️ Skip Pipeline and Load Latest File"):
    with st.spinner("Loading latest RCA file..."):
        latest_file_path = get_latest_csv_filename(data_path + input_folder)
        if not latest_file_path:
            st.error("❌ No RCA file found.")
            st.stop()

        latest_file_name = os.path.basename(latest_file_path)
        df = pd.read_csv(latest_file_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unwanted Unnamed cols

        if 'Feedback' not in df.columns:
            df['Feedback'] = None

        st.session_state.df = df

        st.session_state.latest_file_name = latest_file_name  # Save for use later
        # st.success(f"✅ Loaded file: {latest_file_name}")
        # st.success(f"Feedback Counts: {df['Feedback'].nunique()}")



latest_file_path = get_latest_csv_filename(os.path.join( data_path, input_folder))
latest_file_name = os.path.basename(latest_file_path)

# st.success(f'Global file:>> {latest_file_path}')
# --- Only proceed if df exists ---
if "df" not in st.session_state:

    df = pd.read_csv( latest_file_path)
    st.session_state.df = df
    st.warning("⚠️ Please run or skip the pipeline to load data.")
    st.stop()

df = st.session_state.df
# st.success(f'global df:>> {df.shape}')
# st.success(f"Feedback Counts:>> \n{df['Feedback'].nunique()}")


# Continue with your logic:
df = df[df['Code'].notnull()]

if 'Feedback' not in df.columns:
    df['Feedback'] = None




print(df.columns)
print(df.shape)
print('V1: drilldown of exception works correct but not for remaining')



if st.button("🔄 Reset"):
    st.session_state.level = "exception"
    st.session_state.selected_exception = None
    st.session_state.selected_service = None
    st.rerun()


if "level" not in st.session_state:
    st.session_state.level = "exception"


if st.session_state.level == "exception":
    st.subheader("🚨 Exception Summary")
    exception_df = df[["exception", "exception_count"]].drop_duplicates()

    for _, row in exception_df.iterrows():
        col1, col2, col3 = st.columns([4, 1, 2])
        with col1:
            st.markdown(row['exception'])
        with col2:
            st.markdown(f"Counts: {row['exception_count']}")
        with col3:
            if st.button("Select", key=f"exception_{row['exception']}"):
                st.session_state.selected_exception = row['exception']
                st.session_state.level = "service"
                st.rerun()


elif st.session_state.level == "service":
    st.subheader(f"🧰 Services for Exception: {st.session_state.selected_exception}")
    service_df = df[df['exception'] == st.session_state.selected_exception][["Service", "service_count"]].drop_duplicates()

    for _, row in service_df.iterrows():
        col1, col2, col3 = st.columns([4, 1, 2])
        with col1:
            st.markdown(row['Service'])
        with col2:
            st.markdown(f"Counts: {row['service_count']}")
        with col3:
            if st.button("Select", key=f"service_{row['Service']}"):
                st.session_state.selected_service = row['Service']
                st.session_state.level = "details"
                st.rerun()




    st.button("🔙 Back to Exceptions", on_click=lambda: st.session_state.update(level="exception"))


elif st.session_state.level == "details":
    st.subheader(f"📩 Messages for Service: {st.session_state.selected_service}")
    detail_df = df[
        (df['exception'] == st.session_state.selected_exception) &
        (df['Service'] == st.session_state.selected_service)
    ][["message", "message_count", "latest_error_message", "RCA_and_Soln"]].drop_duplicates()

    for i, row in detail_df.iterrows():
        with st.expander(f"💬 Message (Count: {row['message_count']})"):
            st.markdown("**Message:**")
            st.code(row["message"], language="text")

            st.markdown(f"**Latest Error:** {row['latest_error_message']}")
            st.code(row['latest_error_message'], language="json")
            if st.button("📖 View RCA and Solution", key=f"rca_button_{i}"):
                st.session_state.selected_rca = row["RCA_and_Soln"]
                st.session_state.level = "rca_view"
                st.rerun()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("🔙 Back to Services", on_click=lambda: st.session_state.update(level="service"))
    with col2:
        st.button("🏠 Back to Exceptions", on_click=lambda: st.session_state.update(level="exception"))



elif st.session_state.level == "rca_view":
    st.subheader("🧾 Detailed RCA and Solution")

    content = st.session_state.selected_rca

    # Show RCA content
    with st.container():
        st.markdown("---")
        st.markdown("#### 📄 RCA and Solution Report")
        st.markdown(content, unsafe_allow_html=False)
        st.markdown("---")

    # Feedback input
    st.markdown("### ✍️ Provide Feedback")
    feedback_text = st.text_area("What feedback do you have for this RCA?", key="feedback_input")

    if st.button("💾 Submit Feedback"):
        # Update the feedback in df for the matching row
        idx = df[
            (df['exception'] == st.session_state.selected_exception) &
            (df['Service'] == st.session_state.selected_service) &
            (df['RCA_and_Soln'] == content)
        ].index

        if not idx.empty:
            df.loc[idx, 'Feedback'] = feedback_text
            st.session_state.df = df  # Update session state
            df.to_csv(latest_file_path, index=False)
            st.success("✅ Feedback saved.")
        else:
            st.error("❌ Could not find the record to save feedback.")

    # Unified button row
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🔙 Back to Details"):
            st.session_state.level = "details"
            st.rerun()

    with col2:
        if st.button("🏠 Back to Exceptions"):
            st.session_state.level = "exception"
            st.rerun()

    with col3:
        if st.button("🚀 Launch"):
            if "df" in st.session_state and "latest_file_name" in st.session_state:
                launch_output_path = os.path.join(data_path + output_folder, st.session_state.latest_file_name)
                st.session_state.df.to_csv(launch_output_path, index=False)
                st.success(f"✅ Final data saved to: `{launch_output_path}`")
            else:
                st.error("⚠️ Could not save. Missing data or file name.")


print('latest file name => ',latest_file_path)
st.success(f"Latest saved file{data_path + input_folder + latest_file_name}")
df.to_csv( data_path + 'Test_Results/6_RCA/' + latest_file_name)


