import sys
import time
import requests
import subprocess
import os
from pathlib import Path

def wait_for_api(url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                print("✅ API is healthy")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("❌ API failed to start within timeout")
    return False

def test_endpoints(base_url):
    print("\n🔍 Testing Endpoints...")
    
    # 1. Stats
    try:
        response = requests.get(f"{base_url}/api/stats")
        stats = response.json()
        print(f"📊 Stats matched: Documents={stats.get('documents')}, Chunks={stats.get('chunks')}")
        if stats.get('documents') > 0 and stats.get('chunks') > 0:
            print("✅ Database is populated")
        else:
            print("❌ Database appears empty")
    except Exception as e:
        print(f"❌ Failed to fetch stats: {e}")

    # 2. Query
    query = "What are the main risk factors for Tesla?"
    print(f"\n❓ Querying: '{query}'")
    try:
        start = time.time()
        response = requests.post(f"{base_url}/api/query", json={"question": query})
        data = response.json()
        duration = time.time() - start
        
        if response.status_code == 200:
            print(f"✅ Query successful ({duration:.2f}s)")
            print(f"   Answer: {data.get('answer')[:100]}...")
            print(f"   Sources: {len(data.get('sources'))}")
            print(f"   Confidence: {data.get('confidence'):.2f}")
        else:
            print(f"❌ Query failed: {response.text}")
    except Exception as e:
        print(f"❌ Query exception: {e}")

def main():
    # Start API in background
    print("🚀 Starting API...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(Path(__file__).parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        if wait_for_api("http://localhost:8000"):
            test_endpoints("http://localhost:8000")
        else:
            # Print logs if failed
            print("API Logs:")
            print(api_process.stderr.read().decode())
    finally:
        print("\n🛑 Stopping API...")
        api_process.terminate()
        api_process.wait()

if __name__ == "__main__":
    main()
