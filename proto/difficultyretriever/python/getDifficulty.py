import requests

def get_difficulty():
    url = "http://xenblocks.io/difficulty"
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        json_response = response.json()
        return json_response.get("difficulty", "No difficulty key found")
    else:
        raise RuntimeError("Failed to get the difficulty")

try:
    difficulty = get_difficulty()
    print("Difficulty:", difficulty)
except Exception as e:
    print(f"Error: {e}")
