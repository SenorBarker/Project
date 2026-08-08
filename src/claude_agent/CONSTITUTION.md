Editorial Constitution  
Rules the planning agent follows for paper edits. Every decision must cite a rule ID; otherwise mark it a judgment call, or NEEDS VISUAL REVIEW if it needs assesment.

0. Beat archetypes - closed set of 8  `archetype` must be exactly one of the values below, verbatim. Never invent a new value (e.g. HUMAN, CONTEXT, DEVELOPMENT, OBSERVATION are all invalid), never paraphrase, never use a free-text content-type label.
    Use one of these for every beat. Do not invent new ones
    - `ESTABLISHER` — pre-incident normal state, sets the scene before disruption.
    - `EVENT_TRIGGER` — the precipitating action that leads into the incident
        (an argument escalating, a dog breaking loose, a car pulling out without
        stopping, a hand reaching for the till). May repeat for multi-stage
        incidents.
    - `EVENT_ACTION` — the core physical incident itself, that satisfies the essence of the query . May repeat. highest priority to include 
    - `AFTERMATH` — resolution, reporting, aftermath state.
    - `SOUNDBITE` — audio-lead. the visuals are not important.
    - `MAP` — a synthetic map/recon asset request (interior or geographic).
    - `METRIC` — a chart/stat asset request.
    
1. Picture Editing  
    R1.1 [FOOTAGE] — No jump cuts on the same subject; change angle by 30°+ or shot size.  
    R1.2 [FOOTAGE] — Cut on action.  
    R1.3 [FOOTAGE] — Keep screen direction consistent unless bridged.  
    R1.4 [FOOTAGE] — Match eyeline and position or flag a break.  
    R1.5 [FOOTAGE] — Establish a location before tighter coverage, unless the cold open withholds it.  
    R1.6 - Primary cut points are before and after beats start and end.
    R1.7 - A beat can contain multiple clips
    R1.8 - Merge clips into one if they are from the same camera and there is either overlap, no gap, or only a short pause between them.
    R1.9 — Do not propose beats from the same camera, that overlap in time
    R1.10 -  Video is just as important as audio. If an audio cut splits useable video, then adjust the cut point
    R1.11 — Before finalizing a real segment's frame range, check the transcript for the same span (including gaps between tracked sequences): if dialogue discusses the subject there, add a beat or extend the segment rather than dropping it because tracking data alone was discontinuous.
     
2. Sound Editing  
    R2.1 — Do not cut mid-beat: a cut point must not fall inside a word, sentence, or coherent exchange with usable audio. Move the cut to the nearest boundary (R4.1).
    R2.2 - Audio that is not relevent to the subject of a beat doesn't belong in the beat and should be editied out, as long as R1.9 is satisfied
    R2.3 [FOOTAGE] — Keep ambience under every cut; no hard silence.  
    R2.4 [FOOTAGE] — Prefer L-cuts/J-cuts over hard sound cuts.  
    R2.5 [FOOTAGE] — Match loudness across sources.  
    R2.6 [FOOTAGE] — Music must not punctuate factual changes.
    
3. Sourcing & Balance  
    R3.1 [IGNORE] — Contested claims need two sources or a single-source flag.  
    R3.2 — No statistic without source and timeframe.  
    R3.3 — Include dissent or flag its absence.
    
4. Soundbite Selection  
    R4.1 - Sync that discusses the subject is important and is a priority
    R4.2 — Prefer specific over general.  
    R4.3 — Use the shortest sufficient quote.  
    R4.4 — Do not change meaning by cutting.  
    R4.5 — One idea per beat.
    R4.6 - Ideas must be completed. e.g a a question receives an answer
    
5. Metrics & Map Usage  
    R5.1 —   
    R5.2 — Use a map only if geography matters. spatial questions like "where...", "how far" can be considered geographical questions.  
    R5.3 —   
    R5.4 — RECON_CAM_POSES/RECON_3D need true multi-viewpoint coverage (multiple cameras or moving camera);  — flag [JUDGMENT CALL] if unstated. CUT3R_RECON/VGGT_O_RECON can work from a statrionary pan — only require that the description states the wearer/camera moved or turned during the span.
    
6. Structure & Pacing  
    R6.1 — Do not open with out-of-order soundbites, hooks or teasers
    R6.2 — Open with stakes or a concrete scene in under 10 seconds. This doesn't need to come from the first 10s of the video.  
    R6.3 — Keep one throughline - which answers the user's prompt.  
    R6.4 — Cut to information density, not rhythm.  
    R6.5 — Separate data beats with other content  
    R6.6 — Maps cannot open unless geography is the story.  
    R6.7 — Never end on a map.
        
7. Decision Reporting  
    R7.1 — Name  rejected alternatives and why.  
    R7.2 — Mark unresolved choices as [JUDGMENT CALL].
    R7.3 — Report if the transcript/narration makes a factual claim (e.g. subject identity) that the deterministic tracking/mask data contradicts or cannot confirm. State that the fact is [CONTESTED] and present the contradiction
    
8. Graphics
    R8.1 — Captions must not cover the subject/action and must stay in safe area. The agent only decides whether a caption belongs and what it says; placement and line count are computed.

9. Editorial
    R9.1 — Every beat must have a strong reason to be included. If the answer to the brief will not change if you remove the beat, then remove it 

10. Analysis
    R10.1 —  Audio claims should be backed-up by video proof. Request Mask overlays to identfy key people and objects if they are critical to resolve the prompt. flag: [NEEDS VISUAL REVIEW] on this beat
    R10.2 — Use a metric only if it adds something new.
    R10.3 — Metrics support claims; they do not replace sourcing.
    R10.4 — Physics-based analysis (speed, distance, etc) requires depth, so you must demand 3D recon to solve this
    R10.5 — Metrics can and will support the answer to the prompt. use many, but they must satisfy R10.2 and R10.3
   
Draft v0.4 — edit freely. Keep IDs stable.