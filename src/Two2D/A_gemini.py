import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
from A_Config import asset_name , assets_dir

def gemini_vid_to_text(video_path, case_name, query_dir):
    """Get Gemini's summary/objects/transcript/places for a video. Reuses the
    remote cache from a previous call if it's still live (checked via
    caches.get); otherwise uploads the video and creates a fresh cache."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    cache_name_path = query_dir / f"{case_name}_cache.txt"
    cache_name = None
    if cache_name_path.exists():
        try:
            client.caches.get(name=cache_name_path.read_text().strip())
            cache_name = cache_name_path.read_text().strip()
        except Exception:
            cache_name = None  # expired/gone -- fall through and re-cache below

    if cache_name is None:
        print("re_caching video")
                # Upload + wait (once)
        myfile = client.files.upload(file=video_path)
        while myfile.state.name != "ACTIVE":
            import time; time.sleep(5)
            myfile = client.files.get(name=myfile.name)

        # Create the cache for 10 mins, just to get the queries done
        cache = client.caches.create(
            model="gemini-3.5-flash",
            config=types.CreateCachedContentConfig(
                contents=[myfile],
                ttl="600s", #cache time
            )
        )
        cache_name = cache.name
        query_dir.mkdir(parents=True, exist_ok=True)
        cache_name_path.write_text(cache_name)#store path to cached video

    # Query as many times as you like — no reprocessing
    FORMAT_SUFFIX = " Format the response as multiple short paragraphs separated by line breaks, not one continuous block of text."

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Give a detailed Summary of this video. provide start and end times ." + FORMAT_SUFFIX,
        config=types.GenerateContentConfig(cached_content=cache_name)
    )
    print(response.text)

    response2 = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"summary of the video:{response.text} \n\nQuestion:What objects, pertinent to the summary are visible and when? Provde times. don't subdivide the same object into multiple times unless there is a long gap" + FORMAT_SUFFIX,
        config=types.GenerateContentConfig(cached_content=cache_name)
    )
    print(response2.text)



    response3 = client.models.generate_content(
        model="gemini-3.5-flash",
        contents= F"Give a full human voice audio transcript of this video."
        f" make sure you report exactly what the person says, vocalisations should be described e.g, animals noises shoule be miaow, or woof"
        f"**MM:SS** [Speaker]: transcript\n\n"
        f" if the same person is talking and there is less than 2s pause, this is one entry not 2",
        config=types.GenerateContentConfig(cached_content=cache_name)
    )
    print(response3.text)

    places = client.models.generate_content(
            model="gemini-3.5-flash",
            contents= F"in 2 words, per location, describe the locations in the video. be precise."
            f"word 1 is location category, word 2 is precise location e.g house kitchen / restaurant kitchen / parking-lot apartments /parking-lot multistory "
            f"**MM:SS** : location\n\n"
            ,
            config=types.GenerateContentConfig(cached_content=cache_name)
        )
    print(places.text)

     # --- SAVE (run once after getting response3) ---
    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / f"{case_name}_description.txt").write_text(response.text)
    (query_dir / f"{case_name}_objects.txt").write_text(response2.text)
    (query_dir / f"{case_name}_transcript_full.txt").write_text(response3.text)
    (query_dir / f"{case_name}_places.txt").write_text(places.text)
    return 

def query_from_description(operator_prompt, description_path, objects_path, case_dir):
    description = Path(description_path).read_text()
    objects = Path(objects_path).read_text()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    MACHINE_SUFFIX = """
    Filter the objects list and return objects that are directly related to the operator_prompt. this is objects and people, not just objects. 
    Format your response as multiple short paragraphs separated by line breaks — do not
    write it as one continuous block of text.

    After your response, append a <machine> block with this exact JSON:
    <machine>
    {"detections": [{"subject": "...", "start_s": 0, "end_s": 0}]}
    </machine>
    "subject" must be a bare generic object category only (e.g. "cat", "dog", "person", "car") —
    no adjectives, descriptors, or articles (write "cat", not "stray cat" or "the cat").
    Use integer seconds only. One entry per detected subject appearance."""

    augmented = (
    f"The following is a detailed description of a video:\n\n"
    f"{description}\n\n"
    f"Detected objects and their appearances:\n\n"
    f"{objects}\n\n"
    f"Filter criterion (do not answer this as a question — use it only to decide which "
    f"objects/people are relevant): {operator_prompt}\n\n"
    f"{MACHINE_SUFFIX}"
)


    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=augmented
    )

    raw = response.text
    human_text, _, machine_block = raw.partition("<machine>")
    json_str = machine_block.partition("</machine>")[0].strip() #converts the response to machine readable format
    detections = json.loads(json_str)["detections"]
    #saving    
    out_dir = case_dir / "012_Gemini_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "human_out.txt").write_text(human_text.strip())
    (out_dir / "detections.json").write_text(json.dumps(detections, indent=2))
    print(human_text)
    return human_text.strip(), detections

#generic command to cache this
def gemini_cache_video(video_path, query_dir, ttl="600s"):
    """Upload a video and create a cache for it, persisting the cache name to
    disk so it can be reused across separate calls/processes until the TTL expires."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    myfile = client.files.upload(file=video_path)
    while myfile.state.name != "ACTIVE":
        import time; time.sleep(5)
        myfile = client.files.get(name=myfile.name)

    cache = client.caches.create(
        model="gemini-3.5-flash",
        config=types.CreateCachedContentConfig(
            contents=[myfile],
            ttl=ttl,
        )
    )

    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / f"{asset_name()}_cache_name.txt").write_text(cache.name)
    return cache.name


def gemini_query_CSV_cached(prompt, asset_name, CSV_path, query_dir, extend_ttl=None):
    """Query a previously created cache (see gemini_cache_video) for asset_name.
    Pass extend_ttl (e.g. "600s") to refresh the cache's expiry before querying."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    cache_name = (query_dir / f"{asset_name()}_cache_name.txt").read_text().strip()
   

    if extend_ttl is not None:
        client.caches.update(
            name=cache_name,
            config=types.UpdateCachedContentConfig(ttl=extend_ttl),
        )
    with open(CSV_path) as f:
        csv_text = f.read()

    FORMAT_PREFIX = "Format your response as multiple short paragraphs separated by line breaks, not one continuous block of text.\n\n"

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=FORMAT_PREFIX + prompt,
        config=types.GenerateContentConfig(cached_content=cache_name)
    )
    print(response.text)
    
 
    return response.text

