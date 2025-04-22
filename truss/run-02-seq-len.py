import argparse
import json
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

            all_chunks = []

            for chunk in response.iter_content(chunk_size=4096):
                # chunk may be multiple line-separated JSON dicts
                try:
                    chunk_dict = json.loads(chunk)
                    all_chunks.append(chunk_dict)
                except:
                    continue

            return all_chunks

    eval_df = pd.read_csv("/workspace/analyses/2025-04-22_001-ttft-by-seq-len/rime-eval-sentences.csv")

    warmup_df = eval_df.sample(n=10, random_state=0)

    for data in tqdm(warmup_df.itertuples(), total=len(warmup_df), desc="Running warm up ..."):
        _ = hit_baseten(data.utt_text)

    results_dicts = []

    for data in tqdm(eval_df.itertuples(), total=len(eval_df), desc="Running eval ..."):
        all_results = hit_baseten(data.utt_text)

        results_dict = { "utt_id" : data.utt_id }
        # Only keep first and last token stats for now
        results_dict.update(all_results[0])
        results_dict.update(all_results[-1])

        results_dicts.append(results_dict)

    results_df = pd.DataFrame(results_dicts)

    results_df.to_csv("/workspace/analyses/2025-04-22_001-ttft-by-seq-len/2025-04-22_001-ttft-by-seq-len.csv", index=False)
