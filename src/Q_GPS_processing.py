'''GPS extraction from source recordings - one function per device/app, since
each embeds (or omits) location metadata differently.'''

import json
import re
import subprocess
import pandas as pd
import requests
from A_Config import report_path
from C_CSV_report import add_to_report

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

#ISO 6709: signed lat, signed lon, optional signed altitude, trailing slash
#e.g. "+38.6757+015.8951/"
_ISO6709_RE = re.compile(r"([+-]\d+\.\d+)([+-]\d+\.\d+)(?:([+-]\d+\.?\d*))?/?")


def android_movie_GPS(video_path, api_key):
    '''INPUT  : path to an Android-recorded video file, Google Geocoding API key,
                report CSV path.
       OUTPUT : (lat, lon) floats, parsed from the container's embedded ISO 6709
                "location" tag (e.g. "+38.6757+015.8951/").

                This is a single static fix the OS stamps into the file - not a
                GPS track - so it can only anchor the whole clip to one point.
                Confirmed to be the fix at recording start, so it pairs with
                that clip's frame 0.

                Also silently reverse-geocodes this fix into the report CSV's
                "place" row -- best-effort, never raises, since a failed
                geocode shouldn't break the (lat, lon) this function exists for.

       Returns (None, None) if no location tag is present, or it doesn't parse.'''
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_entries", "format_tags=location,format_tags=location-eng", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    tags = json.loads(probe.stdout)["format"].get("tags", {})
    location = tags.get("location") or tags.get("location-eng")
    if location is None:
        print(f"No location metadata in {video_path}")
        return None, None

    match = _ISO6709_RE.match(location)
    if not match:
        print(f"Unrecognised location format: {location!r}")
        return None, None
    lat, lon, _alt = match.groups()
    lat, lon = float(lat), float(lon)

    try:
        reverse_geocode(lat, lon, api_key)
    except Exception as e:
        print(f"Reverse geocode failed silently: {e}")

    return lat, lon

def reverse_geocode(lat, lon, api_key):
    '''INPUT  : lat, lon floats, Google Geocoding API key, report CSV path.
       OUTPUT : short place name (the first "locality" component, e.g. "Tropea"),
                falling back to the full formatted address if no locality is
                present. Returns None if the lookup fails or finds nothing.
       This is the video's own location (where it was filmed), not tied to any
       one subject event -- also appends it straight to the report CSV.'''
    response = requests.get(
        GEOCODE_URL,
        params={"latlng": f"{lat},{lon}", "key": api_key},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "OK" or not data.get("results"):
        print(f"Reverse geocode failed for ({lat}, {lon}): {data.get('status')}")
        return None

    place = data["results"][0]["formatted_address"]
    for component in data["results"][0]["address_components"]:
        if "locality" in component["types"]:
            place = component["long_name"]
            break

    add_to_report( {"place": place})
    return place


def csv_to_GPS_dict(csv_path):
    import os
    df = pd.read_csv(csv_path)
    return {int(os.path.splitext(str(row["Frame"]))[0]): (row["lat"], row["lon"]) for _, row in df.iterrows()}



