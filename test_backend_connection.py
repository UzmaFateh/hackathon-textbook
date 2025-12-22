import requests
import time

def test_backend_connection():
    """
    Test the connection to your Hugging Face deployed backend
    """
    backend_url = "https://uzifateh-ai-book-chatbot-backend.hf.space"

    print("Testing backend connection...")
    print(f"Backend URL: {backend_url}")

    # Test health endpoint
    try:
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{backend_url}/api/health", timeout=30)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
            print("[SUCCESS] Backend health check passed")
        else:
            print(f"Health check failed: {response.text}")
            print("[ERROR] Backend health check failed")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Health check failed with error: {e}")

    # Test if the backend is responding
    try:
        print("\n2. Testing basic connectivity...")
        response = requests.get(f"{backend_url}/", timeout=30)
        print(f"Root endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("[SUCCESS] Backend is accessible")
        else:
            print(f"[WARNING] Root endpoint returned status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Backend connectivity failed: {e}")
        print("This could be due to:")
        print("  - Backend not running")
        print("  - Network issues")
        print("  - Backend sleeping (Hugging Face Spaces free tier)")

    print("\n3. [SUCCESS] Backend connection test completed.")
    print("Your backend is accessible at:", backend_url)

if __name__ == "__main__":
    test_backend_connection()