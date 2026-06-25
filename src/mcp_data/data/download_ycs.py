"""
Download datasets from S3 for Lightning Studio / local work
Fetch all input data from AWS S3 and place it in the local directory (env variable LOCALDATA_PATH) as parquet files 
"""

import os
from pathlib import Path
from datetime import date, timedelta, datetime
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.client import BaseClient
import argparse

def download_s3_prefix(
    bucket: str,
    prefix: str,
    local_dir: str | Path = ".",
) -> int:
    """Recursively download all objects under an S3 prefix to a local directory.
    Equivalent to::
        aws s3 cp s3://{bucket}/{prefix} {local_dir} --recursive
    """
    s3 = boto3.client("s3")
    local_dir = Path(local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    prefix = prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    downloaded = 0
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel_path = key[len(prefix) :] if prefix else key
                dest = local_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))
                downloaded += 1
                print(f"download: s3://{bucket}/{key} -> {dest}")
    except NoCredentialsError:
        raise SystemExit("AWS credentials not found. Configure ~/.aws/credentials.") from None
    except ClientError as e:
        raise SystemExit(f"S3 download failed: {e}") from e

    print(f"Downloaded {downloaded} object(s) to {local_dir}")
    return downloaded

def ensure_directory_exists(path: str) -> None:
    """Ensure the directory exists, creating it if necessary."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _download_prefix(
    s3_client: BaseClient, bucket_name: str, prefix: str, local_folder: str
) -> int:
    """Download all objects under prefix to local folder.
    Returns:
        Number of files downloaded.
    """
    LOCALDATA_PATH = os.getenv("LOCALDATA_PATH",None)
    if LOCALDATA_PATH is None:
        raise ValueError("LOCALDATA_PATH environment variable is not set")
    count = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            local_path = f"{LOCALDATA_PATH}/{local_folder}/{filename}"
            ensure_directory_exists(local_path)
            print("Downloading to: %s", local_path)
            s3_client.download_file(bucket_name, key, local_path)
            count += 1
    return count


def main(date: str) -> None:
    
    LOCALDATA_PAR_FOLDER = os.getenv("LOCALDATA_PAR_FOLDER","par_curve")
    LOCALDATA_ZERO_COUPON_FOLDER = os.getenv("LOCALDATA_ZERO_COUPON_FOLDER","zero_coupon")
    LOCALDATA_SPOT_FX_FOLDER = os.getenv("LOCALDATA_SPOT_FX_FOLDER","spot_fx_rates")
    LOCALDATA_BUCKET_NAME = os.getenv("LOCALDATA_BUCKET_NAME",None)    
    if LOCALDATA_BUCKET_NAME is None:
        raise ValueError("LOCALDATA_BUCKET_NAME environment variable is not set")    
    LOCALSOURCE_NAME = os.getenv("LOCALSOURCE_NAME",None)
    if LOCALSOURCE_NAME is None:
        raise ValueError("LOCALSOURCE_NAME environment variable is not set")
    
    print("Fetch S3 data for date: %s from bucket: %s", date, LOCALDATA_BUCKET_NAME)

    s3_client = boto3.client("s3")

    par_prefix = f"{LOCALSOURCE_NAME}/{date}/transformed/par"
    par_count = _download_prefix(s3_client, LOCALDATA_BUCKET_NAME, par_prefix, LOCALDATA_PAR_FOLDER)
    if par_count == 0:
        print("No transformed par curve data found for date: %s", date)
    else:
        print("Downloaded %d par curve files for date: %s", par_count, date)

    zero_coupon_prefix = f"{LOCALSOURCE_NAME}/{date}/transformed/zero_coupon"
    zero_coupon_count = _download_prefix(
        s3_client, LOCALDATA_BUCKET_NAME, zero_coupon_prefix, LOCALDATA_ZERO_COUPON_FOLDER
    )
    if zero_coupon_count == 0:
        print("No transformed zero coupon data found for date: %s", date)
    else:
        print(
            "Downloaded %d zero coupon files for date: %s", zero_coupon_count, date
        )

    fx_prefix = f"{LOCALSOURCE_NAME}/{date}/transformed/spot_fx_rates"
    fx_count = _download_prefix(s3_client, LOCALDATA_BUCKET_NAME, fx_prefix, LOCALDATA_SPOT_FX_FOLDER)
    if fx_count == 0:
        print("No transformed spot fx rates data found for date: %s", date)
    else:
        print("Downloaded %d spot fx rates files for date: %s", fx_count, date)

def last_n_fridays(d: date, n: int = 0) -> date:
    # Find the Friday of the week containing d, then go back 7 days.
    # Python: Monday=0 ... Sunday=6, Friday=4
    days_since_friday = (d.weekday() - 4) % 7
    last_friday = d - timedelta(days=days_since_friday)
    return last_friday + timedelta(days=7*n)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    print("Fetch S3 data to local data directory")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--date", type=str, required=True)
    # args = parser.parse_args()
    # main(date="2026-05-22")
    last_friday = last_n_fridays(date.today(), 0)
    LOCALDATA_BUCKET_NAME = os.getenv("LOCALDATA_BUCKET_NAME",None)    
    if LOCALDATA_BUCKET_NAME is None:
        raise ValueError("LOCALDATA_BUCKET_NAME environment variable is not set")    
    LOCALSOURCE_NAME = os.getenv("LOCALSOURCE_NAME",None)
    if LOCALSOURCE_NAME is None:
        raise ValueError("LOCALSOURCE_NAME environment variable is not set")

    download_s3_prefix(
        bucket=LOCALDATA_BUCKET_NAME,
        prefix=f"{LOCALSOURCE_NAME}/{last_friday.strftime('%Y-%m-%d')}/transformed/",
        local_dir=os.getenv("LOCALDATA_PATH","."),
    )
