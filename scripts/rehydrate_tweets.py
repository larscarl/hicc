# SPDX-License-Identifier: MIT

import os
import re
import sys
import time
import argparse
import pandas as pd
from dotenv import load_dotenv

import tweepy
from tweepy.errors import TooManyRequests

BATCH_SIZE = 100  # Twitter v2 get_tweets allows up to 100 IDs per call

ID_REGEX = re.compile(r"/status(?:es)?/(\d+)")


def extract_tweet_id(url: str):
    if not isinstance(url, str):
        return None
    m = ID_REGEX.search(url)
    return m.group(1) if m else None


def main(inp, outp):
    load_dotenv(override=True)
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        print("ERROR: TWITTER_BEARER_TOKEN not set (env or .env).", file=sys.stderr)
        sys.exit(1)

    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)

    # If an output file already exists, continue from it; otherwise start from input
    if os.path.exists(outp):
        df = pd.read_csv(outp, dtype=str, keep_default_na=False)
        print(f"Loaded existing output with {len(df)} rows to resume.")
    else:
        df = pd.read_csv(inp, dtype=str, keep_default_na=False)
        print(f"Loaded input with {len(df)} rows.")

    # Ensure helper columns exist
    if "msg" not in df.columns:
        df["msg"] = ""

    # Compute/refresh tweet_id column (not saved in final file)
    df["_tweet_id"] = df["x_url"].apply(extract_tweet_id)

    # Worklist: ID present AND msg is NaN or empty string
    needs_mask = df["_tweet_id"].notna() & (df["msg"].isna() | (df["msg"] == ""))
    needs = df.loc[needs_mask]
    todo_ids = sorted({tid for tid in needs["_tweet_id"]})
    print(f"Tweets to fetch: {len(todo_ids)}")

    for i in range(0, len(todo_ids), BATCH_SIZE):
        batch = todo_ids[i : i + BATCH_SIZE]
        attempt = 0
        while True:
            try:
                resp = client.get_tweets(ids=batch, tweet_fields=["text"])
                break
            except TooManyRequests as e:
                # Basic backoff if wait_on_rate_limit didn't catch it
                reset = 60
                print(f"Rate limited. Sleeping {reset}s…", flush=True)
                time.sleep(reset)
                attempt += 1
                if attempt > 5:
                    raise
            except Exception as e:
                print(
                    f"Unexpected error on batch starting at {i}: {e}", file=sys.stderr
                )
                # Mark these as not found to avoid infinite loops; continue
                resp = None
                break

        id_to_text = {}
        if resp and resp.data:
            for tw in resp.data:
                # tw.id can be int or str depending on tweepy version
                tid = str(getattr(tw, "id"))
                id_to_text[tid] = tw.text

        # Fill texts for found tweets
        mask_batch = df["_tweet_id"].isin(batch) & (
            df["msg"].astype(str).str.len() == 0
        )
        # Use map via a temporary Series
        to_fill = df.loc[mask_batch, "_tweet_id"].map(id_to_text).fillna("")
        df.loc[mask_batch, "msg"] = to_fill.values

        # Checkpoint after each batch
        df.drop(columns=["_tweet_id"], inplace=True)
        df.to_csv(outp, index=False)
        # Recreate helper col for the next loop
        df["_tweet_id"] = df["x_url"].apply(extract_tweet_id)

        done = (i + BATCH_SIZE) if (i + BATCH_SIZE) < len(todo_ids) else len(todo_ids)
        print(f"Processed {done}/{len(todo_ids)}")

    # Final write (ensures helper column is removed)
    if "_tweet_id" in df.columns:
        df.drop(columns=["_tweet_id"], inplace=True)
    df.to_csv(outp, index=False)
    print(f"Done. Wrote {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rehydrate tweet texts into a CSV.")
    ap.add_argument("input_csv", help="Path to input CSV (expects column 'x_url').")
    ap.add_argument(
        "output_csv",
        nargs="?",
        default="dataset_rehydrated.csv",
        help="Path to output CSV (default: dataset_rehydrated.csv).",
    )
    args = ap.parse_args()
    main(args.input_csv, args.output_csv)
