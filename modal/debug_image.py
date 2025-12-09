import modal
import os

def download_model():
    import esm
    esm.pretrained.esm2_t33_650M_UR50D()

# Exact same image definition as embed_cloud.py
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "fair-esm", "numpy", "pandas", "tqdm")
    .run_function(download_model)
    .add_local_dir(
        "data", 
        remote_path="/root/data",
        ignore=["embeddings", "processed", "__pycache__", "*.pkl", "*.npy"]
    )
)

app = modal.App("debug-image")

@app.function(image=image)
def list_local_files():
    print("Listing /root/data recursively:")
    if not os.path.exists("/root/data"):
        print("ERROR: /root/data does not exist!")
        return
        
    for root, dirs, files in os.walk("/root/data"):
        for name in files:
            print(os.path.join(root, name))

@app.local_entrypoint()
def main():
    list_local_files.remote()
