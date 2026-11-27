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
import json
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
OISST = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv"
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
    """Right whale calls per UTC hour. Hours where the buoy listened for less than
    `min_listen_s` seconds become NaN (missing, not zero); partly heard hours are
    scaled to a full hour (calls x 3600 / seconds listened)."""
    g = tally.set_index("time").sort_index().resample("h")
    h = pd.DataFrame({"calls": g.right.sum().astype(float), "listen_s": g.duration_s.sum()})
    h.loc[h.listen_s < min_listen_s, "calls"] = np.nan
    part = h.listen_s.between(min_listen_s, 3599)
    h.loc[part, "calls"] = (h.calls[part] * 3600 / h.listen_s[part]).round(1)
    # the newest hour is usually still filling in (the page posts 15-min rows): drop it unless complete
    return h.loc[:h.index[h.listen_s >= 3600][-1]]


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


def get_sst(start, lat, lon, previous_csv):
    """MUR SST; if that ERDDAP dataset is down (it often answers 503), NOAA OISST
    (0.25 deg, ~2 weeks behind); if both fail, the SST already in the previous CSV."""
    try:
        return fetch_sst(start, lat, lon)
    except (OSError, ValueError) as e:
        print(f"  MUR SST failed ({e}); trying NOAA OISST")
    try:
        url = (f"{OISST}?sst%5B({start:%Y-%m-%d}T12:00:00Z):1:(last)%5D%5B(0.0)%5D%5B({lat})%5D%5B({lon})%5D")
        df = pd.read_csv(StringIO(get(url)), skiprows=[1])
        return pd.Series(df.sst.to_numpy(), index=pd.to_datetime(df.time, utc=True).dt.floor("D"))
    except (OSError, ValueError) as e:
        print(f"  OISST failed too ({e}); keeping the SST from {previous_csv}")
    old = pd.read_csv(previous_csv, parse_dates=["timestamp"])      # FileNotFoundError: no SST at all, stop
    return old.set_index(old.timestamp.dt.floor("D")).sst_c.groupby(level=0).mean()


def source_note(platform, h):
    return {"name": "WHOI Robots4Whales live buoy detections",
            "note": (f"Real right whale up-calls from the WHOI near real-time buoy {platform.split('/')[0]} "
                     f"(Cape Cod Bay by default), {h.index[0]:%d %b} - {h.index[-1]:%d %b %Y %H:%M} UTC, "
                     "counted every 15 minutes by the buoy's automatic detector (Baumgartner lab, WHOI; many periods "
                     "not yet analyst-reviewed) and refreshed hourly. Hours the buoy was not listening are missing, "
                     "not zero. SST: NOAA MUR via ERDDAP.")}


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
    sst = get_sst(h.index[0], args.lat, args.lon, args.out)
    h["sst_c"] = sst.reindex(h.index.floor("D")).to_numpy()
    h["sst_c"] = h.sst_c.ffill().bfill().round(2)   # SST lags 1-14 days: hold the last value

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(args.out, index_label="timestamp")
    # plain-language source note read by site/build_site.py
    Path(args.out).with_suffix(".source.json").write_text(json.dumps(source_note(args.platform, h), indent=1),
                                                          encoding="utf-8")
    print(f"Wrote {args.out}  ({h.index[0]:%Y-%m-%d %H:%M} to {h.index[-1]:%Y-%m-%d %H:%M} UTC)")
    print(f"Next:  python run_all.py --skip-detector --hourly-csv {args.out}")


if __name__ == "__main__":
    main()
