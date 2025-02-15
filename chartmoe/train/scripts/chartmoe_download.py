import huggingface_hub
import time

def download():
    try:
        huggingface_hub.snapshot_download("IDEA-FinAI/chartmoe", local_dir="./ckpt/chartmoe",resume_download=True,max_workers=4)
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
