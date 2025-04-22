import argparse
import requests

import pandas as pd

from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("api_key")
    args = parser.parse_args()

    def hit_baseten(text):
        headers = {"Authorization": f"Api-Key {args.api_key}"}
        payload = {"text" : text }

        with requests.post(args.url, headers=headers, json=payload, stream=True) as response:

            response.raise_for_status()

            all_chunks = [ chunk for chunk in response.iter_content(chunk_size=4096)]

            return all_chunks

    eval_df = pd.read_csv("/workspace/analyses/2025-04-22_001-ttft-by-seq-len/rime-eval-sentences.csv")

    warmup_df = eval_df.sample(n=10, random_state=0)

    for data in tqdm(warmup_df.itertuples(), total=len(warmup_df), desc="Running warm up ..."):
        _ = hit_baseten(data.utt_text)

    for data in tqdm(eval_df.itertuples(), total=len(eval_df), desc="Running eval ..."):
        all_results = hit_baseten(data.utt_text)
