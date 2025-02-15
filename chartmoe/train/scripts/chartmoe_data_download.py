import huggingface_hub
import time

def download():
    try:
        huggingface_hub.snapshot_download("Coobiw/ChartMoE-Data", local_dir="./data/", repo_type="dataset", resume_download=True)
        return True
    except:
        print("Caught an exception! Retrying...")
        return False

while True:
    result = download()
    if result:
        print("success")
        break  # Exit the loop if the function ran successfully
    time.sleep(1)  # Wait for 1 second before retrying
