"""Real right whale data from NOAA's public Google Cloud bucket -> data/buoy_hourly.csv

    python fetch_dclde.py                       # download (once, ~336 MB, cached) and build the hourly series
    python run_all.py --skip-detector --hourly-csv data/buoy_hourly.csv

Source: DCLDE 2027 "Acoustic Detections and Aerial Surveys of North Atlantic Right
Whales in Cape Cod Bay, MA" (Garcia & Tolkova et al. 2025), NOAA NCEI passive
acoustic archive, gs://noaa-passive-bioacoustic/dclde/2027/. Automated right whale
up-call detections (Raven selection tables, one per day) from 5 MARU recorders,
17 Feb - 12 Jun 2019. Archived data, not live. SST: NOAA OISST v2.1 daily via ERDDAP.
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import fetch_whoi   # reuse its HTTP helper and SST fetcher

GCS = "https://storage.googleapis.com"
PREFIX = "dclde/2027/dclde2027_narw_detections_in_cape_cod_bay/"
BUCKET = "noaa-passive-bioacoustic"
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "dclde_ccb"
START, END = "2019-02-17T17:56:37Z", "2019-06-12T15:26:30Z"   # recording window from the dataset metadata
# channel -> (lat, lon), from the dataset metadata
CHANNELS = {1: (41.9337, -70.2286), 2: (41.9466, -70.4002), 3: (41.8413, -70.4693), 4: (41.8397, -70.3148),
            5: (41.8949, -70.3592)}
FILE_T0 = re.compile(r"_(\d{8})_(\d{6})Z")


def download(sub="data/"):
    """Download every file under PREFIX+sub that is not cached yet; returns local paths."""
    import json
    url = f"{GCS}/storage/v1/b/{BUCKET}/o?prefix={PREFIX}{sub}&maxResults=1000"
    items = json.loads(fetch_whoi.get(url))["items"]
    out = []
    for i, it in enumerate(items):
        dst = CACHE / it["name"][len(PREFIX):]
        if not dst.exists() or dst.stat().st_size != int(it["size"]):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(f"{GCS}/{BUCKET}/{it['name']}", tmp)
            tmp.replace(dst)                         # never leave a half file under the real name
        out.append(dst)
        print(f"\r  {sub} {i + 1}/{len(items)}", end="", flush=True)
    print()
    return out


def read_selections(path):
    """One Raven selection table -> DataFrame(time [UTC], channel, score)."""
    df = pd.read_csv(path, sep="\t", usecols=["Channel", "Begin File", "File Offset (s)", "Score"])
    t0 = df["Begin File"].str.extract(FILE_T0).agg("".join, axis=1)
    time = pd.to_datetime(t0, format="%Y%m%d%H%M%S", utc=True) + pd.to_timedelta(df["File Offset (s)"], unit="s")
    return pd.DataFrame({"time": time, "channel": df["Channel"], "score": df["Score"]})


def to_hourly(det, channel, min_score):
    """Up-calls per UTC hour on one recorder (one channel = one buoy, so calls heard
    on several recorders are not counted twice). Hours outside the recording window are dropped."""
    d = det[(det.channel == channel) & (det.score >= min_score)]
    idx = pd.date_range(pd.Timestamp(START).ceil("h"), pd.Timestamp(END).floor("h") - pd.Timedelta("1h"), freq="h")
    calls = d.set_index("time").resample("h").size().reindex(idx, fill_value=0)
    return pd.DataFrame({"calls": calls.astype(float)}, index=idx)


def fetch_sst_oisst(start, end, lat=41.875, lon=-70.375):
    """Daily NOAA OISST v2.1 (0.25 deg) for 2019, centre of Cape Cod Bay. Used instead of MUR,
    whose ERDDAP server does not answer for 2019 dates within minutes."""
    from io import StringIO
    url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv?sst"
           f"%5B({start:%Y-%m-%d}T12:00:00Z):1:({end:%Y-%m-%d}T12:00:00Z)%5D%5B(0.0)%5D%5B({lat})%5D%5B({lon})%5D")
    with urllib.request.urlopen(url, timeout=300) as r:       # this server is slow (~1 min), not broken
        df = pd.read_csv(StringIO(r.read().decode()), skiprows=[1])
    return pd.Series(df.sst.to_numpy(), index=pd.to_datetime(df.time, utc=True).dt.floor("D"))


def main():
    ap = argparse.ArgumentParser(description="Real Cape Cod Bay right whale detections (NOAA, 2019) -> hourly CSV")
    ap.add_argument("--channel", type=int, default=1, choices=sorted(CHANNELS), help="recorder to use")
    ap.add_argument("--min-score", type=float, default=0.75,
                    help="detector score cut-off; 0.75 is the paper's choice (precision 0.857, recall 0.896)")
    ap.add_argument("--out", default=str(ROOT / "data" / "buoy_hourly.csv"))
    args = ap.parse_args()

    print(f"Downloading from {GCS}/{BUCKET}/{PREFIX} (cached in {CACHE}) ...")
    files = download("data/")
    det = pd.concat([read_selections(f) for f in sorted(files)], ignore_index=True)
    h = to_hourly(det, args.channel, args.min_score)
    print("Fetching 2019 sea-surface temperature from NOAA ERDDAP (about a minute) ...")
    sst = fetch_sst_oisst(h.index[0], h.index[-1])
    h["sst_c"] = sst.reindex(h.index.floor("D")).to_numpy()
    h["sst_c"] = h.sst_c.ffill().bfill().round(2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(args.out, index_label="timestamp")
    # plain-language source note read by site/build_site.py
    Path(args.out).with_suffix(".source.json").write_text(json.dumps({
        "name": "NOAA DCLDE 2027 Cape Cod Bay right whale detections",
        "location": {"place": "Cape Cod Bay, Massachusetts, USA (North Atlantic)",
                     "station": f"seafloor recorder {args.channel} of 5", "lat": CHANNELS[args.channel][0],
                     "lon": CHANNELS[args.channel][1], "approx": False, "recorders": list(CHANNELS.values())},
        "note": (f"Real right whale up-calls recorded in Cape Cod Bay, Massachusetts, {h.index[0]:%d %b} - "
                 f"{h.index[-1]:%d %b %Y}, from one of five seafloor recorders (Garcia & Tolkova et al. 2025, "
                 "NOAA NCEI passive acoustic archive). Counts come from the dataset's own automated up-call detector "
                 f"(score >= {args.min_score}: precision 0.857, recall 0.896); EchoGuard's MiniRocket detector is "
                 "evaluated separately on Kaggle clips.")}, indent=1), encoding="utf-8")
    print(f"{len(det)} detections on all channels; channel {args.channel}, score >= {args.min_score}: "
          f"{int(h.calls.sum())} calls in {len(h)} hours ({h.index[0]:%d %b} - {h.index[-1]:%d %b %Y})")
    print(f"Wrote {args.out}\nNext:  python run_all.py --skip-detector --hourly-csv {args.out} --detector-rates 0.896 0.857")


if __name__ == "__main__":
    main()
