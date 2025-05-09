import boto3
import sys
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
load_dotenv()
import time
from crewai import LLM
from datetime import datetime, timedelta
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError
from langchain_community.callbacks import get_openai_callback
print("Improved log searching and error extraction in Run query")
# Load AWS credentials
load_dotenv()



lite_model = 'gpt-3.5-turbo'
heavy_model = 'gpt-4.1'
# dt_now = datetime.datetime.now
#
# file_name = f"{dt_now}_AI_debugging.csv"

latest_filename = sys.argv[3]

print("File Name:", latest_filename)


import os
import glob

def get_latest_csv_filename(folder_path: str, pattern: str = "*_AI_debugging.csv"):
    files = glob.glob(os.path.join(folder_path, pattern))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return os.path.basename(latest_file)  # Just the file name







print('Phase 1] Extractor')

STATUS_FILE = "/home/rohitsaswadkar/Documents/Projects/AI_debug_Analyser/Test_Results/pipeline_status.txt"
def update_status(msg):
    with open(STATUS_FILE, "w") as f:
        f.write(msg)



# AWS Configuration
config = Config(
    retries={'max_attempts': 10, 'mode': 'adaptive'},
    connect_timeout=60,
    read_timeout=120
)

client = boto3.client('logs', config=config)

def list_log_groups():
    log_groups = []
    next_token = None
    while True:
        try:
            if next_token:
                response = client.describe_log_groups(nextToken=next_token)
            else:
                response = client.describe_log_groups()
            log_groups.extend(group['logGroupName'] for group in response['logGroups'])
            next_token = response.get('nextToken')
            if not next_token:
                break
        except (BotoCoreError, ClientError) as e:
            print(f"⚠️ Error retrieving log groups: {e}")
            break
    return log_groups

#, @operation , @level, @type
query_string = """
fields @timestamp, @message, @service , @operation , @level, @type, @stackTrace
| filter @message like /ERROR/
| sort @timestamp asc
"""



def run_query(log_group, start_time, end_time):
    """Run CloudWatch Insights Query with retries"""
    try:
        query_response = client.start_query(
            logGroupName=log_group,
            startTime=int(start_time * 1000),
            endTime=int(end_time * 1000),
            queryString=query_string
        )
        query_id = query_response["queryId"]
        print(f"🔍 Query started for {log_group} (ID: {query_id})...")

        retries = 0
        while retries < 10:
            try:
                response = client.get_query_results(queryId=query_id)
                if response["status"] == "Complete":
                    logs = response["results"]
                    if logs:
                        print(f"❗ Errors found in {log_group}!")
                    else:
                        print(f"✅ No errors found in {log_group}!")
                    return logs
            except EndpointConnectionError:
                print("⚠️ Connection error, retrying...")
                time.sleep(2 ** retries)
                retries += 1
            time.sleep(2)
        print("⚠️ Query timed out.")
    except (BotoCoreError, ClientError) as e:
        print(f"⚠️ Query failed for {log_group}: {e}")
    return []




def get_time_window_from_dates(start_date_str: str, end_date_str: str):
    start_time = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp())
    end_time = int(datetime.strptime(end_date_str, "%Y-%m-%d").timestamp())
    return start_time, end_time




if __name__ == "__main__":
    save_folder = '1_Merging'
    base_path = os.getenv("data_path") #"/home/rohitsaswadkar/Documents/Projects/AI_debug_Analyser/Test_Results"
    final_csv_path = os.path.join(base_path, save_folder, latest_filename)
    os.makedirs(os.path.join(base_path, save_folder), exist_ok=True)

    # Accept start and end dates from CLI, fallback to default (yesterday to today)
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]  # e.g., "2025-05-03"
        end_date = sys.argv[2]    # e.g., "2025-05-04"
    else:
        today = datetime.now()
        start_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

    start_time, end_time = get_time_window_from_dates(start_date, end_date)

    log_groups = list_log_groups()
    all_logs = []

    for i, log_group in enumerate(log_groups):
        update_status(f"Processing {i + 1}/{len(log_groups)}: {log_group}")
        logs = run_query(log_group, start_time, end_time)

        service_name = '-'.join(log_group.split("/")[-1].split('-')[:-1])
        if logs:
            for log in logs:
                timestamp = next((f["value"] for f in log if f["field"] == "@timestamp"), "N/A")
                message = next((f["value"] for f in log if f["field"] == "@message"), "N/A")
                all_logs.append([log_group, service_name, timestamp, message])

    # Write to CSV (if needed)
    if all_logs:
        import csv
        with open(final_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            print("1] Merging Saved in ", final_csv_path)
            writer.writerow(["LogGroup", "Service", "Timestamp", "Message"])
            writer.writerows(all_logs)








print('Phase 2] Preprocessing')

print('Improved llms reasoning for self identified exception')

data_path = os.getenv("data_path")
# file_name = 'Sample_short_data.csv'
# file_name = 'log_group_trials.csv'
input_folder = '1_Merging/'
output_folder = '2_Transformation/'


folder_path = data_path + input_folder # "/home/rohitsaswadkar/Documents/Projects/AI_debug_Analyser/Test_Results/1_Merging"





# output_path = data_path + output_folder
# csv_path = data_path + input_folder + file_name

data = folder_path + latest_filename
df = pd.read_csv(data)
print("2] Data path for Processing", data)

print('DF', df)
print('Columns', df.columns)
print('DF Shape', df.shape)


def extract_multiple_logs_to_df(df):
    from crewai import Agent, Crew, Task
    error_logs = df['Message']
    service = df['Service']
    Log_group = df['LogGroup']
    Timestamp = df['Timestamp']
    extractor_agent = Agent(
        role="Senior Software Engineer with 30 years of experience in debugging and AI systems",
        goal="Analyze error logs and extract detailed debugging information",
        backstory=(
            "With decades of experience in AI development and deep system debugging, "
            "this agent excels at identifying exceptions, root causes, and helpful insights "
            "from complex error logs for fast issue resolution."
        ),
        verbose=False,
        allow_delegation=False,
        llm = LLM(model = lite_model)
    )

    results = []

    for idx, log in enumerate(error_logs):
        try:
            print(f"🚀 Processing log #{idx + 1}...")

            task = Task(
                description=(
                    "You are given an error log in **stringified JSON format**: {error_log}\n\n"
                    "Your task is to strictly extract the following 6 fields from this JSON **only**. "
                    "Do **not** guess or fabricate values. Only use what is explicitly present in the log.\n\n"

                    "You must analyze the error log completely to extract these fields accurately:\n"
                    # "- **timestamp**: Extract from the 'timestamp' key.\n"
                    "- **Exception**: Analyze the error log and tell which exception is raised in the error log.\n"  # #"- **Service**: Identify the service from the log content.\n"

                    "- **Message**: The specific error message associated with the exception.\n"
                    "- **File**: The file path where the error occurred (may be in the 'file' key or inside the stack trace).\n"
                    "- **Operation**: Extract from the 'operation' or 'op' key.\n"
                    "- **StackTrace**: Extract the full stack trace if available. If not present, return 'No StackTrace'."
                    "- **requestId**: Extract teh requestId from the error log."
                    "⚠️ Return the output strictly as a **Python list with exactly 6 elements in this order**:\n"
                    "[ Exception, Message, File, Operation, StackTrace, requestId]\n\n"

                    "⚠️ If a field is missing, return it as 'No FieldName' — the field name should match the ones below *exactly* in case and spelling:"
                    "'Exception', 'Message', 'File', 'Operation', 'StackTrace', 'requestId' and should be in double quotes."

                    "⚠️ Do not return: 'not available', 'None', 'null', empty string, or any variation.\n"
                    "⚠️ Your answer must use this exact phrasing: 'No **field name' — nothing else and should be in double quotes."


                    "🛠️ Additional Instructions:\n"
                    "- Replace all single quotes with double quotes in extracted values.\n"
                    "- If a value already has double quotes, leave them as is.\n"
                    "- If any string has unmatched or incorrect parentheses/brackets, correct them before returning.\n"
                    "- Do not surround the list or values with markdown or text — return only the raw Python list."

                )
                ,
                expected_output="[ Exception, Message, File, Operation, StackTrace, requestId]",
                agent=extractor_agent,
                async_execution=False
            )
            error_log = {'error_log': log}
            crew = Crew(agents=[extractor_agent], tasks=[task], verbose=False)
            result = crew.kickoff(inputs=error_log)

            results.append(str(result))  # store as a single stringified list

        except Exception as e:
            print(f"❌ Error on log #{idx + 1}: {e}")
            results.append(f"Error: {e}")

    return pd.DataFrame({
        "ErrorMessage": error_logs,
        "RawAgentOutput": results,
        "Service": service,
        "LogGroup": Log_group,
        "Timestamp": Timestamp
    })


import pandas as pd
import ast


def parse_agent_output(agent_output: str) -> list:
    try:
        agent_output = agent_output.strip()
        if agent_output.startswith("[") and agent_output.endswith("]"):
            agent_output = agent_output[1:-1]

        cleaned_output = f"[{agent_output}]"
        parsed = ast.literal_eval(cleaned_output)

        if isinstance(parsed, list):
            while len(parsed) < 6:
                parsed.append(None)
            return parsed[:6]
    except Exception as e:
        print(f"Failed to parse agent output: {e}")

    return [None] * 6


def process_chunk(chunk, columns):
    """Process a single chunk of the DataFrame"""
    chunk['RawAgentOutput'] = chunk['RawAgentOutput'].apply(lambda x: str(x) if isinstance(x, dict) else x)
    chunk[columns] = chunk['RawAgentOutput'].apply(lambda x: pd.Series(parse_agent_output(x)))
    return chunk[['ErrorMessage', 'RawAgentOutput', 'Service', 'LogGroup','Timestamp'] + columns] # static unprocessed columns


def process_df_in_chunks(dfx, output_path, chunk_size=5):
    """Process and save DataFrame in chunks"""
    columns = [ 'exception', 'message', 'file', 'operation', 'stack_trace', 'requestId']  # extracted columns by the Agent

    print('DFX before preprocessing', dfx['Service'].head())

    processed_chunks = []


    total_chunks = len(dfx) // chunk_size + (1 if len(dfx) % chunk_size != 0 else 0)

    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk = dfx.iloc[start_idx:end_idx].copy()

        try:
            print(f"Processing chunk {i + 1}/{total_chunks} (rows {start_idx}-{end_idx})")
            processed_chunk = process_chunk(chunk, columns)
            processed_chunks.append(processed_chunk)

            temp_output = output_path.replace('.csv', f'_chunk_{i}.csv')
            processed_chunk.to_csv(temp_output, index=False)
            print(f"Saved chunk {i + 1} to {temp_output}")

        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            # Save whatever we have so far
            if processed_chunks:
                temp_output = output_path.replace('.csv', f'_partial_chunk_{i}.csv')
                chunk.to_csv(temp_output, index=False)
                print(f"Saved partial chunk {i + 1} to {temp_output}")
            continue


    if processed_chunks:
        final_df = pd.concat(processed_chunks, ignore_index=True)
        final_df.to_csv(output_path, index=False)
        print(f"2] Processed Final data saved to {output_path}")
        return final_df
    else:

        print("No chunks were processed successfully")
        return None



csv_filename = 'log_group_trials.csv'
output_path = data_path + output_folder + latest_filename
dfx = extract_multiple_logs_to_df(df)

processed_df = process_df_in_chunks(dfx, output_path, chunk_size=200)










print('Phase 3] Grouping')


# from AI.Sample_DF_Transformer import output_folder

print('Modified for level 4')

input_folder = '2_Transformation/'
path = os.getenv("data_path") #"/home/rohitsaswadkar/Documents/Projects/AI_debug_Analyser/Test_Results/"
data_path = path + input_folder
output_folder = '3_Grouping/'

 # "/home/rohitsaswadkar/Documents/Projects/AI_debug_Analyser/Test_Results/1_Merging"
# latest_file = get_latest_csv_file(data_path)
data = path + input_folder + latest_filename
df = pd.read_csv(data)
print("3] Data path for Grouping", data)





def grouped(df):

    message_grouped = df.groupby(['exception', 'Service', 'message']).agg(
        message_count=('message', 'count'),
        first_error=('Timestamp', 'max')
    ).reset_index()

    first_details = df.sort_values('Timestamp', ascending = False).drop_duplicates(['exception', 'Service', 'message'])[
        ['exception', 'Service', 'message', 'ErrorMessage', 'file', 'operation', 'stack_trace','requestId','LogGroup', 'Timestamp']
    ].rename(columns={'ErrorMessage': 'latest_error_message'})

    service_count = df.groupby(['exception', 'Service']).size().reset_index(name='service_count')

    exception_count = df.groupby('exception').size().reset_index(name='exception_count')

    final_df = message_grouped \
        .merge(service_count, on=['exception', 'Service'], how='left') \
        .merge(exception_count, on='exception', how='left') \
        .merge(first_details, on=['exception', 'Service', 'message'], how='left')

    final_df = final_df.sort_values(
        by=['exception_count', 'service_count', 'message_count'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return final_df

dfx = grouped(df)
print("processed",dfx.head())
print(dfx.columns)
print(dfx)

print('DFX columns', dfx.columns)

dfx = dfx[['exception', 'exception_count', 'Service', 'service_count',
        'message','message_count', 'file', 'operation', 'stack_trace', 'latest_error_message','requestId', 'LogGroup','Timestamp']]

print(dfx.describe())

# dfx= grouped(df)
csv = 'log_group_trials.csv'
dfx.to_csv( path + output_folder + latest_filename )
print('3] Grouping Data saved:', path + output_folder + latest_filename)








print('4] Log fetcher')





print('Phase 4] Github Agent')

import os
import pandas as pd
import requests
from dotenv import load_dotenv
from crewai.tools import tool
from crewai import Agent, Task, Crew
from crewai.crews import CrewOutput

load_dotenv()
print('##System solved from halucination: ')

input_folder = '3_Grouping/'
output_folder = '4_Code_extraction/'
path = os.getenv("data_path") #"/home/rohitsaswadkar/Documents/Projects/AI_debug_Analyser/Test_Results/"
data_path = path + input_folder +  latest_filename #"log_group_trials.csv"

df = pd.read_csv(data_path)
print("4] Data path for Github agent", data_path)

# df = df[19:]
print("DF shape", df.shape)
print('Columns', df.columns)

print('1] Package extraction')
GITHUB_TOKEN = os.getenv('github_token')
OWNER = "concentra-ai"
BRANCH = "master"


def git_path(REPO, OWNER, BRANCH):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    branch_url = f"https://api.github.com/repos/{OWNER}/{REPO}/branches/{BRANCH}"
    branch_res = requests.get(branch_url, headers=headers)
    branch_data = branch_res.json()

    if 'commit' not in branch_data:
        print(f"❌ Failed to get branch info for {REPO}")
        return []

    tree_sha = branch_data["commit"]["commit"]["tree"]["sha"]

    tree_url = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{tree_sha}?recursive=1"
    tree_res = requests.get(tree_url, headers=headers)
    tree_data = tree_res.json()

    file_list = []

    for item in tree_data.get("tree", []):
        full_path = f"{REPO}/{item['path']}"
        if item["type"] == "tree":

            pass
        elif item["type"] == "blob" and full_path.endswith(".py"):
            file_list.append(full_path)

    return file_list


repos = ['grasp-backend', 'mcpo-chat']
my_files: list = []

for repo in repos:
    files = git_path(repo, OWNER, BRANCH)
    my_files.extend(files)


print(my_files)
print(len(my_files))

print("My files", my_files)



@tool("code extractor tool")
def code_extractor(url):
    """Extracts the code from a GitHub repository given the file path. The input must be a full file path."""
    token = os.getenv('github_token')

    headers = {
        "Authorization": f"token {token}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("File fetched!\n")
        return response.text
    else:
        print(f"Failed. Status code: {response.status_code}")
        print("URL tried:", url)
        return f"Error: Failed to fetch file. Status {response.status_code}"


# print(code_extractor('/services/'+ 'ai-chat' + path))


def github_agent(dictionary: dict, my_files: list, repositories: list) -> CrewOutput:
    agent = Agent(
        role="Senior AI developer with mastery in code debugging and error log analysis",
        goal="Analyze the error log and extract correct file paths to fetch code from the GitHub repository.",
        backstory=(
            "With decades of experience in AI development and deep system debugging, "
            "this agent excels at identifying exceptions, root causes, and helpful insights "
            "from complex error logs and extracts the GitHub repository codes using these insights for fast issue resolution."
        ),
        allow_delegation=False,
        verbose=True,
        llm=LLM(model=lite_model)
    )

    task = Task(
        description=(
            "You are an advanced code retrieval agent with the following resources:\n"
            "1. Input dictionary containing: {dictionary} (with 'Service','stack_trace','file').\n"
            "2. Input list containing: {files_list} having the complete github repositories package of all available repositories."
            "2. Available repositories: {repo_list}.\n\n"

            "Your task is to fetch code from GitHub by intelligently constructing the correct file path:\n"

            "REPOSITORY SELECTION RULES:\n"
            "- You have to understand the file path 'file' and also understand the repositories package {files_list}."  # 1 original
            "By understanding the repositories package you will have the idea that which 'file' is in which repository,"
            " and in which folder and will be helpful to create the exact file path."
            "- If service name starts with 'ai': Use 'mcpo-chat' repository.\n"
            "- For all other services: Select appropriate repository from {repo_list}.\n\n"

            "PATH CONSTRUCTION PROTOCOL:\n"
            "1. Pre-process the file path:\n"
            "   - Remove any 'var/task/' segments.\n"
            "   - Ensure path starts with '/'.\n\n"

            "2. Construct full URL by combining:\n"
            "   [Selected Repository Base URL] + [Processed Path]\n\n"

            "ERROR RECOVERY PROCEDURE:\n"
            "1. If initial attempt fails (404/error):\n"
            "   - Analyse the repositories package which will tell as which file is in which folder then try."
            "   - Extract service name from {dictionary}.\n"
            "   - Insert service name after 'services/' in path.\n"
            "   - Example: '/services/<service_name>/processed_path'.\n\n"


            "TOOL USAGE:\n"
            "- Always use code_extractor tool with the constructed path.\n"
            "- Verify the tool's response for success/errors.\n\n"

            "OUTPUT REQUIREMENTS:\n"
            "- Return the successfully fetched code.\n"
            "- If unsuccessful after retries, return:\n"

        ),
        expected_output="The code content of the specified file from GitHub.",
        tools=[code_extractor],
        agent=agent
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff(
        inputs={'dictionary': dictionary, "repo_list": repositories, "files_list": my_files})  # Pass the path directly
    return result


# Example usage
# path = "/repos/owner/repo/contents/path/to/file.py"  # Your GitHub file path
# print(github_agent(path, 'ai-chat'))
# df['Service'] = df['Service'].apply(lambda x: "-".join(x.split('_')))
import time

valid_cols = ['Service', 'stack_trace', 'file']

output_folder = '4_Code_extraction/'
path = os.getenv("data_path") #"/home/rohitsaswadkar/Documents/Projects/AI_debug_Analyser/Test_Results/"
save_path = os.path.join(path, output_folder)

os.makedirs(save_path, exist_ok=True)

if 'Code' not in df.columns:
    df['Code'] = None


def service_processor(x):
    if x.startswith('ai'):
        x = '-'.join(x.split('_')[:2])

    return x



df['Service'] = df['Service'].apply(lambda x: service_processor(x))
print(df['Service'])

for i, row in df.iterrows():
    start_time = time.time()
    if row['file'] == 'No File':
        continue
    dictionary = {col: row[col] for col in df.columns if col in valid_cols}

    print(f"\nProcessing row {i}:")
    print(f"  Service: {dictionary.get('Service', 'N/A')}")
    print(f"  File: {dictionary.get('file', 'N/A')}")

    try:

        crew_output = github_agent(dictionary, my_files, repositories=os.getenv('grasp_urls'))

        if hasattr(crew_output, 'raw_output'):
            result = crew_output.raw_output
        elif hasattr(crew_output, 'result'):
            result = crew_output.result
        elif hasattr(crew_output, '__str__'):
            result = str(crew_output)
        else:
            result = "CrewOutput received but couldn't extract content"

        df.loc[i, 'Code'] = result
        print(f"  Result from github_agent: {result}")

    except Exception as e:
        error_msg = f"Error in github_agent: {str(e)}"
        df.loc[i, 'Code'] = error_msg
        print(error_msg)

    time_taken = time.time() - start_time
    print(f"  Time taken for this row: {time_taken:.2f} seconds")

    print("-" * 40)

output_file = os.path.join(save_path, latest_filename)
df.to_csv(output_file, index=False)
print(f"\n4] Github extractor Results saved to: {output_file}")

print("\n" + "=" * 50)
print(f"Completed processing {len(df)} rows")
print("=" * 50 + "\n")









print("Phase 5] RCA Agent")


print('RCA and Solution giving Agent')
import os
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.crews import CrewOutput
import time
load_dotenv()



input_folder = '4_Code_extraction/'
path = os.getenv("data_path") #"/home/rohitsaswadkar/Documents/Projects/AI_debug_Analyser/Test_Results/"
data_path = path + input_folder + latest_filename #"log_group_trials.csv"
output_folder = '6_RCA/'
# os.environ["OPENAI_MODEL_NAME"] = 'gpt-4-turbo'



df = pd.read_csv(data_path)
print("5] Data path for RCA Agent", data_path)
# df = df[20:]
print("DF shape", df.shape)
print('Columns', df.columns)

def RCA_agent(dictionary: dict, web_solution: None, history: None) -> CrewOutput:
    agent = Agent(
        role="Expert AI Developer and Root Cause Analyst",
        goal="Analyze error logs, code, and web search solutions to provide a clear Root Cause Analysis and actionable solutions.",
        backstory=(
            "You are a highly experienced AI Developer specializing in system debugging, "
            "error investigation, and fast issue resolution. You have deep expertise in reading error logs, "
            "understanding stack traces, and mapping errors to the correct sections of code. "
            "Your solutions are precise, easy to follow, and help developers quickly fix problems."
        ),
        allow_delegation=False,
        verbose=True,
        llm = LLM(model= heavy_model)
    )

    task = Task(
        description=(
            "You are assigned the role of a Root Cause Analyst and Solution Architect.\n\n"
            "You have the dictionary: {dictionary} and web_solution: {web_solution} and history: {history}."
            "You have the following inputs:\n"
            "- A dictionary {dictionary} containing:\n"
            "    * 'service': The name of the service where the error occurred.\n"
            "    * 'file': The specific file involved.\n"
            "    * 'stack_trace': The stack trace showing the sequence of calls.\n"
            "    * 'error_log': The captured error log.\n"
            "    * 'Code': Snippets of the relevant code.\n"
            "- An optional list called 'web_solution' which contains any external suggestions found online.\n\n"
            "- An optional list of 'history' also provided of the previous solutions and recommendations. "
            "- The history will have the chats of your previous solutions, validator agents feedback on your solutions and the web scrapper agents feedback, "
            "this all will help you to providing the solutions for the 'error_log'."

            "Your objectives:\n"
            "1. Analyze the error_log, stack_trace, and code to deeply understand where and why the error occurred.\n"
            "2. Perform a clear and thorough Root Cause Analysis (RCA) explaining:\n"
            "    - Dont made RCA too lengthy, just pinpoint."
            "    - What caused the issue.\n"
            "    - Where exactly in the code or system it originated (if the error occurred due to the code).\n"
            "    - Why it happened.\n"
            "    - Any contributing factors (wrong input, missing dependency, logic error, etc.).\n\n"

            "3. Propose a **solution** that includes:\n"
            "- If code change is needed: Provide the corrected code snippet and clearly mention where the changes should be made.\n" # with line no
            "- If it's a configuration issue: Explain the steps to fix it.\n"
            "- If it's an external dependency issue: Recommend proper fixes.\n\n"

            "4. Use information from 'web_solution' if available to enhance your answer.\n"
            "5. Your final output must have two clearly separated sections:\n"
            "    - **Root Cause Analysis**\n"
            "    - **Solution**\n\n"

            "Ensure the explanation is clear, structured, and easy for a developer to understand and apply quickly."
        ),
        expected_output=(
            "**Root Cause Analysis:**\n"
            "- [Briefly explain the error: what it is, where it happened, why it happened.]\n"
            "- [Identify the exact code lines/files involved, if possible.]\n"
            "- [Mention any patterns from the stack trace or error log if relevant.]\n\n"
            "**Previous Code:**"
            "- ['If code fix: then provide the previous code where the change is needed.\n"
            "**Solution:**\n"
            "- [If code fix: specify exactly where to make changes and show updated code snippets.]\n"
            "- [If configuration or environment issue: list clear steps to fix.]\n"
            "- [If external dependency/library issue: explain how to resolve.]\n"
            "- [In case multiple possible solutions exist, mention the best recommended one.]\n\n"
            "**Note:** Keep your language simple, instructional, and solution-focused. Make it easy for a developer to directly apply your recommendations."
        ),
        agent=agent
    )



    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff(inputs={'dictionary': dictionary, "web_solution": web_solution, 'history':history})  # Pass the path directly
    token_info = {
        'prompt_tokens': None,
        'completion_tokens': None,
        'total_tokens': None
    }



    with get_openai_callback() as cb:
        result = crew.kickoff(inputs={'dictionary': dictionary, "web_solution": web_solution, 'history': history})
        token_info = {
            'prompt_tokens': cb.prompt_tokens,
            'completion_tokens': cb.completion_tokens,
            'total_tokens': cb.total_tokens,
            'cost_usd': cb.total_cost
        }

    return result.output if hasattr(result, "output") else result, token_info



valid_cols = ['Service', 'file','stack_trace','error_log', 'Code']


print("\n" + "=" * 50)
print("Starting processing of DataFrame rows")
print("=" * 50 + "\n")



df['RCA_and_Soln'] = None
df['Prompt_Tokens'] = None
df['Completion_Tokens'] = None
df['Total_Tokens'] = None
df['Cost_USD'] = None

for i, row in df.iterrows():

    start_time = time.time()

    if row['file'] == 'No File':
        continue

    dictionary = {col: row[col] for col in df.columns if col in valid_cols}


    print(f"\nProcessing row {i}:")
    print(f"  Service: {dictionary.get('Service', 'N/A')}")
    print(f"  File: {dictionary.get('file', 'N/A')}")


    # try:
    #     result = RCA_agent(dictionary, None,None)
    #
    #     df.at[i, 'RCA_and_Soln'] = result
    #     print(f"  Result from github_agent: {result}")
    # except Exception as e:
    #     print(f"  Error in github_agent: {str(e)}")
    try:
        result, token_info = RCA_agent(dictionary, None, None)

        df.at[i, 'RCA_and_Soln'] = result
        df.at[i, 'Prompt_Tokens'] = token_info['prompt_tokens']
        df.at[i, 'Completion_Tokens'] = token_info['completion_tokens']
        df.at[i, 'Total_Tokens'] = token_info['total_tokens']
        df.at[i, 'Cost_USD'] = token_info['cost_usd']

        print(f"Result: {result}")
        print(f"Token Usage: {token_info}")

    except Exception as e:
        print(f"Error: {str(e)}")


df.to_csv(path + output_folder + latest_filename) #'log_group_trials.csv')
print('5] RCA O/P data saved in:', path + output_folder + latest_filename)