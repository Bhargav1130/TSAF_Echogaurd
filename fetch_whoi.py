"""Fetch REAL right whale call counts from a WHOI Robots4Whales buoy, plus NOAA
sea-surface temperature, and write an hourly table for the pipeline:

    python fetch_whoi.py                                   # Cape Cod Bay buoy, last 60 days
    python fetch_whoi.py --platform nybnw2602/nybnw2602_nybnw --lat 40.33 --lon -73.42
    python run_all.py --skip-detector --hourly-csv data/buoy_hourly.csv

Source: DMON/LFDCS near real-time detections, Baumgartner lab, WHOI
(https://robots4whales.whoi.edu). The buoy reports call tallies every 15 min.
These are automatic detections; many periods are not yet analyst-reviewed.
SST: NOAA CoastWatch ERDDAP, MUR SST v4.1 (jplMURSST41), daily.
"""
import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://dcs.whoi.edu"
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv"
# one table row: <a id="YYYYmmddHHMMSS"> (UTC; the visible text is EST) then
# Sei, Fin, Right, Humpback, Other, Duration (listening seconds)
ROW = re.compile(r'<a id="(\d{14})">[^<]*</a></td>' + r'\s*<td[^>]*>(\d*)</td>' * 6)
DAY = re.compile(r'_summary_(\d{8})\.html')
ROOT = Path(__file__).resolve().parent


def get(url, tries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "EchoGuard student project (python urllib)"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("windows-1252", "replace")
        except urllib.error.HTTPError:
            raise                                   # 404 etc.: retrying will not help
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def parse_tally(html):
    """15-min tally rows -> DataFrame(time [UTC], right, duration_s)."""
    rows = [(m[0], m[3] or 0, m[6] or 0) for m in ROW.findall(html)]
    df = pd.DataFrame(rows, columns=["time", "right", "duration_s"])
    df["time"] = pd.to_datetime(df.time, format="%Y%m%d%H%M%S", utc=True)
    return df.astype({"right": int, "duration_s": int})


def to_hourly(tally, min_listen_s=1800):
    """Sum right whale calls per UTC hour. Hours where the buoy listened for
    less than `min_listen_s` seconds become NaN (missing, not zero)."""
    g = tally.set_index("time").sort_index().resample("h")
    h = pd.DataFrame({"calls": g.right.sum().astype(float), "listen_s": g.duration_s.sum()})
    h.loc[h.listen_s < min_listen_s, "calls"] = np.nan
    return h.loc[:h.calls.last_valid_index()]       # drop the unfinished current hour


def fetch_tally(platform, days, cache_dir):
    """Download the last `days` daily tally pages (finished days are cached)."""
    dep, plat = platform.split("/")
    page = f"{BASE}/{dep}/{plat}_html/{plat}_summary"
    today = get(page + ".html")
    dates = sorted(set(DAY.findall(today)))[-(days - 1):] if days > 1 else []
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, d in enumerate(dates):
        f = cache_dir / f"{d}.html"
        if f.exists() and i < len(dates) - 1:       # yesterday may still be filling in: refetch it
            html = f.read_text(encoding="utf-8")
        else:
            html = get(f"{page}_{d}.html")
            f.write_text(html, encoding="utf-8")
            time.sleep(0.5)                         # be polite to the WHOI server
        parts.append(parse_tally(html))
        print(f"\r  day {i + 1}/{len(dates)} ({d})", end="", flush=True)
    print()
    parts.append(parse_tally(today))
    return pd.concat(parts).drop_duplicates("time")


def fetch_sst(start, lat, lon, end=None):
    """Daily MUR SST (deg C) at the buoy position, from `start` to `end` (default: the latest day)."""
    stop = "last" if end is None else f"{end:%Y-%m-%d}T09:00:00Z"
    url = (f"{ERDDAP}?analysed_sst%5B({start:%Y-%m-%d}T09:00:00Z):1:({stop})%5D"
           f"%5B({lat})%5D%5B({lon})%5D")
    df = pd.read_csv(StringIO(get(url)), skiprows=[1])          # row 1 holds units
    return pd.Series(df.analysed_sst.to_numpy(), index=pd.to_datetime(df.time, utc=True).dt.floor("D"))


def main():
    ap = argparse.ArgumentParser(description="Fetch real WHOI buoy right whale detections + NOAA SST")
    ap.add_argument("--platform", default="maccb2606/maccb2606_maccb",
                    help="<deployment>/<platform> as in the dcs.whoi.edu URL (default: Cape Cod Bay buoy)")
    ap.add_argument("--days", type=int, default=60, help="how many recent days to fetch")
    ap.add_argument("--lat", type=float, default=41.95, help="latitude for SST (default: Cape Cod Bay)")
    ap.add_argument("--lon", type=float, default=-70.30, help="longitude for SST (default: Cape Cod Bay)")
    ap.add_argument("--out", default=str(ROOT / "data" / "buoy_hourly.csv"))
    args = ap.parse_args()

    print(f"Fetching {args.platform} from {BASE} ...")
    try:
        tally = fetch_tally(args.platform, args.days, ROOT / "data" / "whoi_cache" / args.platform.replace("/", "_"))
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach {BASE} ({e.reason}). Some networks block or cannot resolve whoi.edu: "
                 "try another network (college Wi-Fi, phone hotspot) or a VPN, then run again.")
    h = to_hourly(tally)
    print(f"{len(tally)} tally rows -> {len(h)} hours, {int(h.calls.isna().sum())} without listening, "
          f"{int(np.nansum(h.calls))} right whale calls")

    print("Fetching SST from NOAA ERDDAP ...")
    sst = fetch_sst(h.index[0], args.lat, args.lon)
    h["sst_c"] = sst.reindex(h.index.floor("D")).to_numpy()
    h["sst_c"] = h.sst_c.ffill().bfill().round(2)   # MUR lags 1-2 days: hold the last value

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(args.out, index_label="timestamp")
    print(f"Wrote {args.out}  ({h.index[0]:%Y-%m-%d %H:%M} to {h.index[-1]:%Y-%m-%d %H:%M} UTC)")
    print(f"Next:  python run_all.py --skip-detector --hourly-csv {args.out}")


if __name__ == "__main__":
    main()
