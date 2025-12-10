import modal
import os

app = modal.App("ls-volume")
vol = modal.Volume.from_name("airr-ml-25-data")

@app.function(volumes={"/data": vol})
def list_files():
    print("Listing /data/data:")
    try:
        items = os.listdir("/data/data")
        for name in items:
            path = os.path.join("/data/data", name)
            if os.path.isdir(path):
                print(f"[DIR]  {name}")
            else:
                print(f"[FILE] {name}")
    except Exception as e:
        print(f"Error: {e}")

@app.local_entrypoint()
def main():
    list_files.remote()
