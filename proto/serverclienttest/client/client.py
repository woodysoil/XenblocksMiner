import requests

def main():
    url = "http://localhost:2357/"
    response = requests.get(url)
    print("Response from server:", response.text)

if __name__ == "__main__":
    main()
