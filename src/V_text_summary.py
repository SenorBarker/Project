


from A_Config import set_case, assets_dir, report_path, asset_name
#turn a video into a text description
from Two2D.A_gemini import gemini_cache_video
with open(report_path()) as f:
        csv_text = f.read()
video_path = assets_dir() / f"{asset_name()}_overlay_video.mp4"

def gemini_final_report():
    #turn a video into a text description. nothing needs to be passed. it's all contained in the script
    event_descrption_prompt = (
            F" Read the CSV first, these are measurements "
            f"'''csv\n{csv_text}\n```\n\n"
        f" The camera wearer and <<subject>>s are the main focus, so base the report on this. Look for green highlights/masks and focus on those if you see them, explictly mention them."
        f"give a 1 sentence exatablisher that gives location, time and date"
        f" then give details on the <<subject>>s"
        f"only mention objects the subject interacts with, or is near"
        f"only mention people that either ineract with, react to, or speak about the subject"
        F"mention who talks to or about the <<subject>>\n\n"
        f"using the CSV fields <<subject_distance>>(distance subject travels) <<subject_direction_degs>> (what direction, 0 is north) talk about the movement of <<subject>> "
        f"using the CSV fields <<subject_first_time>> <<subject_last_time>> talk about the duration of the event the subject does "
            
        )
    transcript_prompt = (
            F"Give a full human voice audio transcript of this video. "
            F" Read the CSV first"
            f"'''csv\n{csv_text}\n```\n\n"
            f"only include audio that is about the <<subject>>, addressed to the <<subject>> or from the <<subject>>mention people that either ineract with, react to, or speak about the subject"
        f" make sure you report exactly what the person says, vocalisations should be described e.g, animals noises shoule be miaow, or woof"
            f"**MM:SS** [Speaker]: transcript\n\n"
            f" if the same person is talking and there is less than 2s pause, this is one entry not 2"     
        )
    from Two2D.A_gemini import gemini_query_CSV_cached
    event_description = gemini_query_CSV_cached(event_descrption_prompt, asset_name,  assets_dir())
    transcript =  gemini_query_CSV_cached(transcript_prompt, asset_name, assets_dir())

    #save and add to report
    (assets_dir() / f"{asset_name()}_event_description.txt").write_text(event_description)
    (assets_dir() / f"{asset_name()}_transcript.txt").write_text(transcript)
    event_descrption_path = assets_dir() / f"{asset_name()}_event_description.txt"
    transcript_path       = assets_dir() / f"{asset_name()}_transcript.txt"
    from C_CSV_report import add_to_report
    report_items = {
        "event_description"      : f"{asset_name()}_event_description.txt", 
        "transcript"         : f"{asset_name()}_transcript.txt",
    }
    add_to_report(report_items)