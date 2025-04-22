import argparse
import requests

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("api_key")
    parser.add_argument("text")
    args = parser.parse_args()

    headers = {"Authorization": f"Api-Key {args.api_key}"}
    payload = {"text" : args.text}

    with requests.post(args.url, headers=headers, json=payload, stream=True) as response:

        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                print(chunk)
