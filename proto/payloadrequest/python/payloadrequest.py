import requests
import json

def main():
    hashed_data = "$argon2id$v=19$m=95400,t=1,p=1$WEVOMTAwODIwMjJYRU4$StL5GRjgWFXtBA5I3A52VZTLwgu+9nQQjoZ715otB45ttpfXEN1R/uu8H4XI8XYS/mA5a5z8PuPsC7adROIc2g" # Replace with your actual data
    key = "c31b1290659cc5d08a00fdd556dc330545486545a3251c3681254db33b9e689179ddbfbaf6a13fafe127097ff1efb646911535d1384239178589065c5bc8c024"
    submitaccount = "..."
    worker_id = "..."

    payload = {
        "hash_to_verify": hashed_data,
        "key": key,
        "account": submitaccount,
        "attempts": "123456",
        "hashes_per_second": "1234",
        "worker": worker_id
    }

    print("Payload:", json.dumps(payload, indent=4))

    try:
        response = requests.post('http://xenblocks.io/verify', json=payload, timeout=10)
        print("Server Response:", response.json())
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

if __name__ == "__main__":
    main()
