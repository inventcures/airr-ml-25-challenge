import modal
import os

app = modal.App("debug-volume")
vol = modal.Volume.from_name("airr-ml-25-data")

@app.function(volumes={"/data": vol})
def list_files():
    print("Listing /data recursively:")
    for root, dirs, files in os.walk("/data"):
        for name in files:
            print(os.path.join(root, name))

@app.local_entrypoint()
def main():
    list_files.remote()
