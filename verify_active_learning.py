
import requests
import pandas as pd
import json
import os
import time

BASE_URL = "http://127.0.0.1:5000"
TEST_FILE = "test_dataset.csv"

def create_test_dataset():
    # Create a dummy CSV if it doesn't exist uploaded
    # For this test, we assume the dataset is already 'uploaded' if we place it in the right dir?
    # Or we use the /upload endpoint.
    
    df = pd.DataFrame({
        'Feature1': range(10),
        'Feature2': range(10, 20),
        'Target': range(20, 30)
    })
    
    # Save a temporary file
    df.to_csv("test_upload.csv", index=False)
    
    files = {'dataset': open('test_upload.csv', 'rb')}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    if response.status_code == 200:
        print(f"✅ Uploaded test dataset: {response.json().get('filename')}")
        return response.json().get('filename')
    else:
        print("❌ Failed to upload dataset")
        return None

def test_mode(mode, filename):
    print(f"\n🧪 Testing Mode: {mode}")
    
    payload = {
        "model": "rf",
        "curiosity": 0.5,
        "dataset_filename": filename,
        "input_columns": ["Feature1", "Feature2"],
        "target_columns": [{"name": "Target", "weight": 1.0, "optimization": "max"}],
        "active_learning_mode": mode,
        "prompt_style": "parameter-format",
        "hybrid_weights": {"w_llm": 0.5, "w_ml": 0.5}
    }
    
    start = time.time()
    try:
        response = requests.post(f"{BASE_URL}/run-experiment", json=payload)
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ {mode} Success ({duration:.2f}s)")
                # Check for expected keys
                if 'results_table' in data:
                     print("   - Results table present")
                if 'tsne_figure' in data:
                     print("   - t-SNE figure present")
            else:
                print(f"❌ {mode} Failed: {data.get('error')}")
        else:
            print(f"❌ {mode} HTTP Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ {mode} Exception: {e}")

if __name__ == "__main__":
    # Ensure server is up (it should be running via flask run in background)
    print("Checking server status...")
    try:
        requests.get(BASE_URL)
    except:
        print("❌ Server not accessible! Is flask run executed?")
        exit(1)

    filename = create_test_dataset()
    if filename:
        test_mode("ML_MODE", filename)
        test_mode("LLM_AGENT_MODE", filename)
        test_mode("HYBRID_MODE", filename)
        
    # clean up
    if os.path.exists("test_upload.csv"):
        os.remove("test_upload.csv")
