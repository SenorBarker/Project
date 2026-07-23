"""
Geolocation.py — standalone module for fetching Google Street View images and metadata, and matching them to video frames via SIFT.

Functions
---------
fetch_metadata(lat, lon, api_key)
    Returns precise location, pano_id, date, and status for the nearest panorama.

fetch_image(lat, lon, heading, api_key, size, fov, pitch)
    Returns a PIL Image of the Street View scene at the given heading.

fetch_candidates(lat, lon, api_key, headings)
    Fetches images + metadata at multiple headings around a point.
    Returns a list of dicts: {"image": PIL.Image, "metadata": dict, "heading": float}
"""

import json
import math
import cv2
import numpy as np
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd

STATIC_URL   = "https://maps.googleapis.com/maps/api/streetview"
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

DATA_DIR = Path(__file__).parent.parent / "Data"


def fetch_metadata(lat: float, lon: float, api_key: str) -> dict:
    """
    Query the Street View Metadata API for the nearest panorama to (lat, lon).

    Returns a dict with keys: status, pano_id, lat, lon, date.
    Raises RuntimeError if the API returns a non-OK status.
    """
    params = {
        "location": f"{lat},{lon}",
        "key": api_key,
    }
    response = requests.get(METADATA_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "OK":
        raise RuntimeError(f"Street View Metadata API returned status: {data.get('status')}")

    return {
        "status":  data["status"],
        "pano_id": data.get("pano_id"),
        "lat":     data["location"]["lat"],
        "lon":     data["location"]["lng"],
        "date":    data.get("date"),
    }


def fetch_image(
    lat: float,
    lon: float,
    heading: float,
    api_key: str,
    size: str = "640x640",
    fov: int = 90,
    pitch: int = 0,
) -> Image.Image:
    """
    Fetch a single Street View static image and return it as a PIL Image.

    heading : compass bearing in degrees (0 = north, 90 = east, etc.)
    fov     : horizontal field of view in degrees (max 120)
    pitch   : vertical angle in degrees (-90 to 90)
    """
    params = {
        "size":     size,
        "location": f"{lat},{lon}",
        "heading":  heading,
        "fov":      fov,
        "pitch":    pitch,
        "key":      api_key,
    }
    response = requests.get(STATIC_URL, params=params, timeout=10)
    response.raise_for_status()

    # The Static API returns a JPEG directly; a grey "no imagery" image has no error status,
    # so we check content-type as a basic guard.
    if "image" not in response.headers.get("Content-Type", ""):
        raise RuntimeError("Street View Static API did not return an image.")

    return Image.open(BytesIO(response.content))


def fetch_pano(
    lat: float,
    lon: float,
    api_key: str,
    headings: tuple = (0, 45, 90, 135, 180, 225, 270, 315),
    case_name: str | None = None,
) -> dict:
    """
    Fetch Street View images at several headings from a single panoramic camera.
    Returns a dict keyed by heading (int degrees):
        {
            0:   {"image": PIL.Image, "metadata": dict},
            45:  {"image": PIL.Image, "metadata": dict},
            ...
        }
    Metadata is fetched once; images are fetched per heading.
    Raises RuntimeError (from fetch_metadata) if no panorama exists at this location.

    If case_name is given, results are cached to/from:
        Data/{case_name}/012_Streetview_panos/{pano_id}_{heading}.jpg
        Data/{case_name}/012_Streetview_panos/{pano_id}_metadata.json
    On subsequent calls images are loaded from disk and the image API is not called.
    """
    metadata = fetch_metadata(lat, lon, api_key)

    #saving
    save_dir = DATA_DIR / case_name / "012_Streetview_panos"
    save_dir.mkdir(parents=True, exist_ok=True)
    pano_id = metadata["pano_id"]
    meta_path = save_dir / f"{pano_id}_metadata.json"

    if not meta_path.exists():
        meta_path.write_text(json.dumps(metadata, indent=2))

    pano = {}
    for h in headings:
        img_path = save_dir / f"{pano_id}_{h}.jpg"
        if img_path.exists():
            image = Image.open(img_path)
        else:
            image = fetch_image(lat, lon, h, api_key)
            image.save(img_path, format="JPEG")
        pano[h] = {"image": image, "metadata": metadata}

    return pano


def show_pano(candidates: list[dict], title: str = "Street View candidates") -> None:
    """
    Display a list of candidate dicts (from fetch_candidates) in a 2-row grid.
    Each panel is labelled with its heading in degrees.
    """
    n = len(candidates)
    cols = (n + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(cols * 4, 8))
    axes = axes.flatten()

    for i, (heading,candidate) in enumerate(candidates.items()):
        axes[i].imshow(candidate["image"])
        axes[i].set_title(f"{heading}°")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def compute_sift(image, mask_watermark: bool = False, use_colour: bool = False) -> tuple:
    """
    Run SIFT on a single image (PIL Image or numpy array).
    mask_watermark=True excludes the bottom 27px (Google Street View logo).
    use_colour=True computes 384-dim descriptors on O1/O2/O3 opponent colour channels THIS DOESN'T WORK (but I've not tested it to make sure it doesn't work becasue it's bad in theory, vs implemented incorrectly.)
    instead of standard 128-dim greyscale. Both sides of any match must use the same setting.
    Returns (keypoints, descriptors).
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = None
    if mask_watermark:
        mask = np.ones(gray.shape, dtype=np.uint8)
        mask[-27:, :] = 0
    sift = cv2.SIFT_create()
    if not use_colour:
        kp, des = sift.detectAndCompute(gray, mask)
        return kp, des
   
    # Opponent colour channels — O1/O2 are chrominance (illumination-invariant), O3 is luminance
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)
    O1 = (R - G) / math.sqrt(2)
    O2 = (R + G - 2 * B) / math.sqrt(6)
    def to_uint8(ch):
        ch = ch - ch.min()
        mx = ch.max()
        if mx > 0:
            ch = ch * (255.0 / mx)
        return np.clip(ch, 0, 255).astype(np.uint8)
    kp = sift.detect(gray, mask)
    _, d1 = sift.compute(to_uint8(O1), kp)
    _, d2 = sift.compute(to_uint8(O2), kp)
    _, d3 = sift.compute(gray, kp)
    des = np.hstack([d1, d2, d3])
   
    return kp, des


def add_sift_to_candidates(candidates: list[dict], use_colour: bool = False) -> list[dict]:
    """
    Run SIFT on each candidate's image and store results back in the dict.
    Adds 'kp' and 'des' keys to each entry in place.
    """
    for candidate in candidates.values():
        kp, des = compute_sift(candidate["image"], mask_watermark=True, use_colour=use_colour)
        candidate["kp"]  = kp
        candidate["des"] = des
    return candidates

'''this may be a raddbit hole - parking for the moment - it's the feature mapping'''
def match_sift(des_a: np.ndarray, des_b: np.ndarray, ratio: float = 0.75) -> list:
    """
    Match two sets of SIFT descriptors using FLANN and Lowe's ratio test.
    Returns a list of good cv2.DMatch objects.
    Returns [] if either descriptor set is None or too small.
    """
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return []

    # FLANN: approximate nearest-neighbour search via KD-tree — much faster than brute force
    # on SIFT's 128-dim descriptors. trees=5, checks=50 is the standard accuracy/speed trade-off.
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50),
    )

    # k=2 → get the two nearest neighbours for each descriptor
    matches = flann.knnMatch(des_a, des_b, k=2)

    # Lowe's ratio test: keep match only if clearly better than the runner-up
    return [m for m, n in matches if m.distance < ratio * n.distance]



'''going to make a 3d recon of the street using the pano'''
def fetch_panos(
    lat: float,
    lon: float,
    api_key: str,
    step_m: float = 15,
    headings: tuple = (0, 45, 90, 135, 180, 225, 270, 315),
    case_name: str | None = None,
) -> list[dict]:
    """
    Fetch up to 5 Street View panoramas — the centre and then one at each of the 4 cardinal coordinates.

    Finds beighbouring panos by trying N/S/E/W offsets of step_m metres 

    Returns separate panos, each a list of 8 heading dicts
    as returned by fetch_pano. Metadata is inside each dict.
    """
    dlat = step_m / 111000
    dlon = step_m / (111000 * math.cos(math.radians(lat)))

    pano_c = fetch_pano(lat, lon, api_key, headings,case_name)
    meta_c = pano_c[0]["metadata"]

    # Try N, S, E, W until we get a different pano
    offsets = [
        (lat + dlat, lon),
        (lat - dlat, lon),
        (lat, lon + dlon),
        (lat, lon - dlon),
    ]

    seen_ids = {meta_c["pano_id"]}
    results = []
    directions = ["N", "S", "E", "W"]
    for (qlat, qlon), direction in zip(offsets, directions):
        try:
            meta = fetch_metadata(qlat, qlon, api_key)
            if meta["pano_id"] not in seen_ids:
                seen_ids.add(meta["pano_id"])
                pano_n = fetch_pano(meta["lat"], meta["lon"], api_key, headings,case_name)
                results.append(pano_n)
            else:
                results.append(None)
        except RuntimeError:
            results.append(None)

    if all(r is None for r in results):
        raise RuntimeError(f"No neighbouring panorama found within {step_m}m — try increasing step_m.")

    pano_n, pano_s, pano_e, pano_w = results
    
    return pano_c, pano_n, pano_s, pano_e, pano_w


#Made this a bit later - this should compare the cross-shaped panos
# it just finds matches, doesn't figure out where they are in the world 
def match_pano_sets(*panos, use_colour: bool = False) -> list[dict]:
    """
    Take up to 5 pano sets (any may be None), run SIFT on all valid ones,
    then match all images vs the centre's images - 64 comparisons per neighbour

    Creates results, which is a dict that contains
    panos - with the iinput panos, plus their sift key points added in
        kp":      list,      # keypoints from pano A image
    Then 
    Returns a dict of matches 
              {
            "heading_c":   float,
            "heading_n":   float
            "matches":   list,      # cv2.DMatch good matches
        }
    """
    results = {}
    
    pano_centre = panos[0]
    neighbours = list(panos[1:])#take the non centres
    if not any(neighbours):
        raise RuntimeError("Need at least one non-None neighbour pano to match against.")

    #we add the new data onto the pano, so they extend with extra data
    add_sift_to_candidates(pano_centre, use_colour=use_colour)
    for pano in neighbours:
        if pano is not None:#if a pano is missing we skip over
            add_sift_to_candidates(pano, use_colour=use_colour)
    #store in results
    results["panos"] = {"c": panos[0], "N": panos[1], "S": panos[2], "E": panos[3], "W": panos[4]}
    #process data
    for neighbour in neighbours:
        if neighbour is None:
            continue
        direction = ["N", "S", "E", "W"][neighbours.index(neighbour)]
        pairs = []
        for hc, centre_data in pano_centre.items(): #items aer key-value pairs - hc and the data
             for hn, neighbour_data in neighbour.items():
                matches = match_sift(centre_data["des"], neighbour_data["des"])
                pairs.append({
                    "heading_c": hc,
                    "heading_n": hn,
                    "matches":   matches,
                })

        results[direction] = {
            "image_comparisons":  pairs,
        }

    return results


def cross_match_table(match_results) -> "pd.DataFrame":
    """
    Pivot match_results (from match_pano_sets) into a readable table.

    Rows    = centre headings
    Columns = neighbour direction x heading combinations
    Values  = number of SIFT matches
    """
    rows = []
    for direction, data in match_results.items():
        if direction == "panos":
            continue
        for r in data["image_comparisons"]:
            rows.append({
                "heading_c": f"{int(r['heading_c'])}°",
                "heading_n": f"{direction}_{int(r['heading_n'])}°",
                "n_matches": len(r["matches"]),
            })

    df = pd.DataFrame(rows)
    return df.pivot(index="heading_c", columns="heading_n", values="n_matches")


def cross_match_table2(match_results) -> "pd.DataFrame":
    """
    Pivot match_results into a matrix without sorting.
    Rows = centre headings, columns = neighbour headings, in loop order.
    """
    data = {}
    cols = []
    for direction, dir_data in match_results.items():
        if direction == "panos":
            continue
        for r in dir_data["image_comparisons"]:
            hc = f"{int(r['heading_c'])}°"
            hn = f"{direction}_{int(r['heading_n'])}°"
            if hc not in data:
                data[hc] = {}
            if hn not in cols:
                cols.append(hn)
            data[hc][hn] = len(r["matches"])

    return pd.DataFrame(data, index=cols).T


def show_matches(results, direction, heading_c, heading_n):
    comparison = next(
        c for c in results[direction]["image_comparisons"]
        if c["heading_c"] == heading_c and c["heading_n"] == heading_n
    )
    img_c = np.array(results["panos"]["c"][heading_c]["image"].convert("RGB"))
    img_n = np.array(results["panos"][direction][heading_n]["image"].convert("RGB"))
    kp_c  = results["panos"]["c"][heading_c]["kp"]
    kp_n  = results["panos"][direction][heading_n]["kp"]
    out = cv2.drawMatches(img_c, kp_c, img_n, kp_n, comparison["matches"], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(16, 6))
    plt.imshow(out)
    plt.axis("off")
    plt.title(f"{direction} | centre {heading_c}° vs neighbour {heading_n}°")
    plt.tight_layout()
    plt.show()


def show_matches_tb(results, direction, heading_c, heading_n):
    from matplotlib.patches import ConnectionPatch
    comparison = next(
        c for c in results[direction]["image_comparisons"]
        if c["heading_c"] == heading_c and c["heading_n"] == heading_n
    )
    img_c = np.array(results["panos"]["c"][heading_c]["image"].convert("RGB"))
    img_n = np.array(results["panos"][direction][heading_n]["image"].convert("RGB"))
    kp_c  = results["panos"]["c"][heading_c]["kp"]
    kp_n  = results["panos"][direction][heading_n]["kp"]
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 12))
    ax_top.imshow(img_c)
    ax_bot.imshow(img_n)
    ax_top.axis("off")
    ax_bot.axis("off")
    for m in comparison["matches"]:
        pt_c = kp_c[m.queryIdx].pt
        pt_n = kp_n[m.trainIdx].pt
        con = ConnectionPatch(
            xyA=pt_n, coordsA=ax_bot.transData,
            xyB=pt_c, coordsB=ax_top.transData,
            color="lime", linewidth=0.5, alpha=0.6,
        )
        fig.add_artist(con)
        ax_top.plot(*pt_c, "o", color="lime", markersize=3)
        ax_bot.plot(*pt_n, "o", color="lime", markersize=3)
    fig.suptitle(f"{direction} | centre {heading_c}° vs neighbour {heading_n}°")
    plt.tight_layout()
    plt.show()



#visualise 2 images and their corresponding feature matches
def show_frame_matches(img1, img2, pts1, pts2, title="", max_lines=50):
    from matplotlib.patches import ConnectionPatch
    if isinstance(img1, (str, Path)):
        img1 = Image.open(img1)
    if isinstance(img2, (str, Path)):
        img2 = Image.open(img2)
    img1 = np.array(img1.convert("RGB"))
    img2 = np.array(img2.convert("RGB"))
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 10))
    ax_top.imshow(img1)
    ax_bot.imshow(img2)
    ax_top.axis("off")
    ax_bot.axis("off")
    step = max(1, len(pts1) // max_lines)
    for p1, p2 in zip(pts1[::step], pts2[::step]):
        con = ConnectionPatch(
            xyA=p2, coordsA=ax_bot.transData,
            xyB=p1, coordsB=ax_top.transData,
            color="lime", linewidth=0.5, alpha=0.6,
        )
        fig.add_artist(con)
        ax_top.plot(*p1, "o", color="lime", markersize=3)
        ax_bot.plot(*p2, "o", color="lime", markersize=3)
    fig.suptitle(title or f"{len(pts1)} correspondences")
    plt.tight_layout()
    plt.show()


def rotation_matrix(h_deg: float) -> np.ndarray:
    h = math.radians(h_deg)
    return np.array([
        [ math.cos(h), 0, -math.sin(h)],
        [ 0,           1,  0           ],
        [ math.sin(h), 0,  math.cos(h) ],
    ])


#create rotations
_rotations = {}
_headings = [0, 45, 90, 135, 180, 225, 270, 315]
for hc in _headings: # loop through headings
    h1 = math.radians(hc)#make radians
    for hn in _headings:
        dh = math.radians(hn - hc) #dh is heading difference
        R = np.array([
            [ math.cos(dh), 0, math.sin(dh)],
            [ 0,            1, 0            ],
            [-math.sin(dh), 0, math.cos(dh)],
        ])
        _rotations[(hc, hn)] =R





def filter_epipolar(match_results: dict, threshold: float = 2.0) -> tuple:
    """
    Filter pano match_results 4*(8 vs 8) by epipolar geometry using known camera positions and headings.
    Calls essential_matrices once per direction, then checks each match against its epipolar line.
    Returns filtered match_results (same dict structure) and a report DataFrame.
    """
    #clone the results so we can edit them - can't just copy becasue of CV2's match data can't be 'pickled'
    filtered_results = {"panos": match_results["panos"]}
    for d in ["N", "S", "E", "W"]:
        if d in match_results:
            filtered_results[d] = {
                "image_comparisons": [
                    {**comp, "matches": list(comp["matches"])}
                    for comp in match_results[d]["image_comparisons"]
                ]
            }
    
    #camerea intrinsics (K)
    f = 320.0 #640x640 with 90 degree angle
    K = np.array([[f, 0, 320.0], [0, f, 320.0], [0, 0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    #make a new dict and add in the panos to keep them safe
    #filtered = {"panos": match_results["panos"]} #It's equivalent to: filtered = {} filtered["panos"] = match_results["panos"]
    report_rows = [] 
    
    #redefine coordinates for centre camera
    lat_c  = match_results["panos"]["c"][0]["metadata"]["lat"]
    lon_c  = match_results["panos"]["c"][0]["metadata"]["lon"]

    for direction in ["N", "S", "E", "W"]: # loop throuhg the 4 panos
        print(f"checking {direction} direction")
        if direction not in match_results or match_results["panos"][direction] is None:
                #skip if empty
            continue

        dir_data = filtered_results[direction]#extract the data we need for this pano vs pano pairing
        #translation calcs that are the same every time - can't complete
        lat_n  = match_results["panos"][direction][0]["metadata"]["lat"]
        lon_n  = match_results["panos"][direction][0]["metadata"]["lon"]        
        dlat_m = (lat_n - lat_c) * 111000
        dlon_m = (lon_n - lon_c) * 111000 * math.cos(math.radians(lat_c))
        print("translation calculated")

        for comparison in dir_data["image_comparisons"]: # then for each image vs image
            print("checking 1 image pair")
            hc = comparison["heading_c"] # define to make easier to read later
            hn = comparison["heading_n"]
            h1 = math.radians(hc) # use radians not degrees
            R = _rotations[(hc, hn)] # obtain the rotation data
            #convert the translation into camera centric movemnent (i.e south can be forward, right, back or left!)
            t = np.array([ 
                    dlon_m * math.cos(h1) - dlat_m * math.sin(h1),
                    0.0,
                    dlon_m * math.sin(h1) + dlat_m * math.cos(h1),
                    ])
            #now turn translation insto a matrix with skew-symmetric matrix (for cross producting)
            #the skew turns the point into a line. 
            tx = np.array([
                        [ 0,    -t[2],  t[1]],
                        [ t[2],  0,    -t[0]],
                        [-t[1],  t[0],  0   ],
                    ])
            #make the matrix
            E = tx @ R # ESSENTIAL
            F = K_inv.T @ E @ K_inv # matrix that turns a pixel xy1 into a line in the other image
            print("transformation matrices made")
            #obtain the key points using key lookups from the dict
            kp_c   = match_results["panos"]["c"][hc]["kp"]
            kp_n   = match_results["panos"][direction][hn]["kp"]

            inlier_indices = []
            for i, match in enumerate(comparison["matches"]):
                pt_c = np.array([*kp_c[match.queryIdx].pt, 1.0])
                pt_n = np.array([*kp_n[match.trainIdx].pt, 1.0])
                line_normal = F @ pt_c
                dist = abs(line_normal @ pt_n) / math.sqrt(line_normal[0]**2 + line_normal[1]**2)
                if dist < threshold:
                    inlier_indices.append(i)#collect indices
            inliers = [comparison["matches"][i] for i in inlier_indices]
            #make a report
            print("making report")
            report_rows.append({
                "direction": direction,
                "heading_c": hc,
                "heading_n": hn,
                "raw":       len(comparison["matches"]),
                "inliers":   len(inliers),
            })

            comparison["matches"] = inliers
            comparison["inlier_count"] = len(inliers)
            print("pair completed")
    return filtered_results, pd.DataFrame(report_rows)

def triangulate_matches(filtered_results: dict) -> list[dict]:
    """
    Triangulate all filtered match pairs into 3D world points.
    World frame: X=East, Y=Up, Z=North, origin at centre camera position.
    Returns a list of {"pt3d": np.array([x,y,z]), "des": np.array(128,)} dicts.
    """
    f = 320.0
    K = np.array([[f, 0, 320.0], [0, f, 320.0], [0, 0, 1.0]], dtype=np.float64)

    lat_c = filtered_results["panos"]["c"][0]["metadata"]["lat"]
    lon_c = filtered_results["panos"]["c"][0]["metadata"]["lon"]

    out = []

    for direction in ["N", "S", "E", "W"]:
        if direction not in filtered_results or filtered_results["panos"][direction] is None:
            continue

        lat_n = filtered_results["panos"][direction][0]["metadata"]["lat"]
        lon_n = filtered_results["panos"][direction][0]["metadata"]["lon"]
        dlat_m = (lat_n - lat_c) * 111000
        dlon_m = (lon_n - lon_c) * 111000 * math.cos(math.radians(lat_c))
        #neighbour camera position x = east/west, y = up/down, z = north south
        C2 = np.array([dlon_m, 0.0, dlat_m])

        for comparison in filtered_results[direction]["image_comparisons"]:
            if not comparison["matches"]:
                continue
            hc = comparison["heading_c"]
            hn = comparison["heading_n"]

            R1 = rotation_matrix(hc)
            R2 = rotation_matrix(hn)
            P1 = K @ np.hstack([R1, np.zeros((3, 1))])
            P2 = K @ np.hstack([R2, (-R2 @ C2).reshape(3, 1)])

            kp_c  = filtered_results["panos"]["c"][hc]["kp"]
            kp_n  = filtered_results["panos"][direction][hn]["kp"]
            des_c = filtered_results["panos"]["c"][hc]["des"]

            #make lists
            pts1 = np.array([kp_c[m.queryIdx].pt for m in comparison["matches"]], dtype=np.float64).T
            pts2 = np.array([kp_n[m.trainIdx].pt for m in comparison["matches"]], dtype=np.float64).T

            pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2) # this does the heavy lifting here
            pts3d = (pts4d[:3] / pts4d[3]).T

            for i, m in enumerate(comparison["matches"]):
                out.append({
                    "pt3d":      pts3d[i], #3d point
                    "des":       des_c[m.queryIdx],#sift descriptor for this feature
                    "kp_c_pt":   kp_c[m.queryIdx].pt,#2d point on centre camerea's image
                    "heading_c": hc, # direction of the photo
                })

    return out


def localise_frame(frame_path, points_3d: list[dict], K_video: np.ndarray = None,
                   lat_c: float = None, lon_c: float = None,
                   reprojection_error: float = 50.0,
                   pitch_deg: float = None,
                   debug: bool = False,
                   panos_c: dict = None,
                   use_colour: bool = False) -> dict:
    """
    Match a video frame against a Street View 3D point database and estimate camera pose.

    frame_path : path to the video frame 
    points_3d  : output of triangulate_matches — list of {"pt3d", "des"} dicts
    K_video    : 3x3 intrinsic matrix for the video camera. If None, approximated
                 from image dimensions — calibrate for accurate results.
    lat_c, lon_c : GPS of the Street View centre camera. If provided, output
                   includes absolute lat/lon in addition to metres.

    Returns {"rvec", "tvec_m", "inliers", "n_matches"} always.
    Adds {"lat", "lon"} if lat_c and lon_c are supplied.
    """
    frame = Image.open(frame_path)
    kp, des = compute_sift(frame, use_colour=use_colour)

    if des is None or len(des) < 4:
        raise RuntimeError("Not enough features in video frame")

    w, h = frame.size

    db_des = np.array([p["des"] for p in points_3d], dtype=np.float32)
    db_pts = np.array([p["pt3d"] for p in points_3d], dtype=np.float64)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw   = flann.knnMatch(des, db_des, k=2)
    good  = [m for m, n in raw if m.distance < 0.75 * n.distance]
    print(f"{len(kp)} keypoints in frame, {len(good)} matches after ratio test")
    
    if len(good) < 4:
        raise RuntimeError(f"Only {len(good)} matches after ratio test — not enough for PnP")

    pts_3d = db_pts[[m.trainIdx for m in good]]
    pts_2d = np.array([kp[m.queryIdx].pt for m in good], dtype=np.float64)

    if K_video is None:
        f_approx = max(h, w)
        K_video = np.array([[f_approx, 0,        w / 2],
                            [0,        f_approx, h / 2],
                            [0,        0,        1    ]], dtype=np.float64)

    rvec_init = np.zeros((3, 1), dtype=np.float64)
    tvec_init = np.zeros((3, 1), dtype=np.float64)
    use_guess = pitch_deg is not None
    if use_guess:
        rvec_init[0, 0] = math.radians(pitch_deg)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K_video, None,
        rvec=rvec_init, tvec=tvec_init,
        useExtrinsicGuess=use_guess,
        reprojectionError=reprojection_error,
    )

    if not success:
        raise RuntimeError("solvePnPRansac failed to find a solution")

    # recover camera position in world frame: C = -R^T @ tvec
    R_mat, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R_mat.T @ tvec).flatten()
    east_m, up_m, north_m = cam_pos
    forward_world = R_mat.T @ np.array([0.0, 0.0, 1.0])
    heading_deg = math.degrees(math.atan2(forward_world[0], forward_world[2])) % 360
    result = {"rvec": rvec, "tvec_raw": tvec,
              "cam_pos_m": {"east": east_m, "up": up_m, "north": north_m},
              "heading_deg": heading_deg,
              "inliers": inliers, "n_matches": len(good)}

    if lat_c is not None and lon_c is not None:
        result["lat"] = lat_c + north_m / 111000
        result["lon"] = lon_c + east_m / (111000 * math.cos(math.radians(lat_c)))

    if debug:
        vis = np.array(frame.convert("RGB"))
        inlier_idx = inliers.flatten()
        pts3d_in = pts_3d[inlier_idx]
        pts2d_in = pts_2d[inlier_idx]
        projected, _ = cv2.projectPoints(pts3d_in, rvec, tvec, K_video, None)
        projected = projected.reshape(-1, 2)
        for p2, proj in zip(pts2d_in, projected):
            cv2.circle(vis, (int(p2[0]),   int(p2[1])),   5, (0, 255, 0), -1)
            cv2.circle(vis, (int(proj[0]), int(proj[1])), 5, (0, 0, 255), -1)
            cv2.line(vis, (int(p2[0]), int(p2[1])), (int(proj[0]), int(proj[1])), (255, 255, 0), 1)
        plt.figure(figsize=(16, 9))
        plt.imshow(vis)
        plt.title(f"{len(good)} matches  |  {len(inlier_idx)} inliers — green=matched 2D  red=reprojected 3D")
        plt.axis("off")
        plt.show()

        if panos_c is not None:
            for h, pano in panos_c.items():
                pts_f = [kp[m.queryIdx].pt for m in good if points_3d[m.trainIdx]["heading_c"] == h]
                pts_s = [points_3d[m.trainIdx]["kp_c_pt"] for m in good if points_3d[m.trainIdx]["heading_c"] == h]
                show_frame_matches(frame, pano["image"], pts_f, pts_s,
                                   title=f"centre {h}°  |  {len(pts_f)} matches")

    return result


def estimate_pitch(frame_path, K_video: np.ndarray, debug: bool = False) -> float:
    """
    Estimate camera pitch from the horizon vanishing point in a street scene.
    Detects roughly horizontal line segments, finds where they converge, and
    computes pitch from the horizon y-position relative to the principal point.
    Returns pitch in radians (negative = pitched down, positive = pitched up).
    """
    frame = Image.open(frame_path)
    img   = np.array(frame.convert("RGB"))
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=100, maxLineGap=10)

    if lines is None:
        raise RuntimeError("No lines detected in frame")

    # keep lines within 30° of horizontal with meaningful horizontal extent
    h_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        #if abs(angle) < 30 and abs(x2 - x1) > 50:
        h_lines.append((x1, y1, x2, y2))

    if len(h_lines) < 2:
        raise RuntimeError(f"Only {len(h_lines)} horizontal lines found — not enough")

    # pairwise intersections
    h, w = gray.shape
    intersections_y = []
    for i in range(len(h_lines)):
        for j in range(i + 1, len(h_lines)):
            x1, y1, x2, y2 = h_lines[i]
            x3, y3, x4, y4 = h_lines[j]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            if -w < ix < 2 * w and -h < iy < h:
                intersections_y.append(iy)

    if not intersections_y:
        raise RuntimeError("No valid line intersections found")

    horizon_y = float(np.median(intersections_y))
    cy = K_video[1, 2]
    fy = K_video[1, 1]
    pitch = math.atan((horizon_y - cy) / fy)

    if debug:
        vis = img.copy()
        for x1, y1, x2, y2 in h_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.line(vis, (0, int(horizon_y)), (w, int(horizon_y)), (0, 0, 255), 2)
        plt.figure(figsize=(16, 9))
        plt.imshow(vis)
        plt.title(f"horizon y={horizon_y:.1f}  pitch={math.degrees(pitch):.2f}°")
        plt.axis("off")
        plt.show()

    return pitch
