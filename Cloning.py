from dotenv import load_dotenv
print("Improved log searching and error extraction in Run query")
load_dotenv()
import subprocess
import os


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Must be set in your system or .env

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in environment variables.")

repos = {
    "grasp-backend": "https://github.com/concentra-ai/grasp-backend.git",
    "mcpo-chat": "https://github.com/concentra-ai/mcpo-chat.git",
    "patient-app": "https://github.com/concentra-ai/patient-app"
}

local_dir = os.path.expanduser("~/PycharmProjects/PythonProject/github_repos_2")
os.makedirs(local_dir, exist_ok=True)

for name, repo_url in repos.items():
    secure_url = repo_url.replace(
        "https://", f"https://<your-username>:{GITHUB_TOKEN}@"
    )
    clone_path = os.path.join(local_dir, name)
    if not os.path.exists(clone_path):
        print(f"Cloning {name} into {clone_path}...")
        subprocess.run(["git", "clone", secure_url, clone_path])
    else:
        print(f"{name} already cloned at {clone_path}")

