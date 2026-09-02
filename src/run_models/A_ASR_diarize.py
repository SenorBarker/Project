"""
Local speech transcript: faster-whisper for the words, pyannote for who said
them -- replacing the Gemini transcript in Two2D/A_gemini_v04._windowed_query.

Gemini was asked for "**MM:SS** [Speaker]: ..." and answered in that shape, but
neither half of it was real: [Speaker] is the literal placeholder from the prompt
(the model was never doing diarization) and the times are the model's guess at
its own clock, which drifts and skips minutes at a time on a 34-minute clip.
Here the times come from Whisper's word-level alignment and the speaker label
from pyannote's segmentation, so both are measured rather than narrated.

The Gemini transcript is left exactly where it is under 012_Gemini_outputs;
this stage writes alongside it in 013_ASR_outputs and nothing overwrites it.

Every function is zero-argument callable -- paths resolve through A_Config off
the active case, so the notebook only needs set_case(...) then
transcribe_and_diarize().

    from A_Config import set_case
    from run_models.A_ASR_diarize import transcribe_and_diarize
    set_case("503_EGO_noodles_sound")
    transcribe_and_diarize()

Outputs, all under 013_ASR_outputs and all named for the *case*, not the
experiment -- this stage reads the source video and nothing an experiment
changes, so it runs once per case and every experiment on that case reads the
same files:
    <case>_audio16k.wav        mono 16 kHz -- what both models actually read
    <case>_asr_words.json      every word with start/end/probability + speaker
    <case>_diarization.rttm    pyannote's turns, standard RTTM
    <case>_audio_events.json   AudioSet tags for the non-speech sound
    <case>_transcript_asr.txt  the readable one: **MM:SS-MM:SS** [SPEAKER_00]: ...
"""

import ctypes
import importlib.util
import json
import os
import subprocess
import types
from pathlib import Path

import torch

# Run as a script (python run_models/A_ASR_diarize.py <case>) only src/run_models
# is on sys.path, and A_Config lives a level up; imported the normal way this is
# already on the path and the insert is a no-op.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from A_Config import asr_dir, asr_transcript_path, case_name, source_video_path

# Whisper: large-v3 fits fp16 on the 3090 with room for pyannote afterwards
# (they are loaded in sequence, not together -- see transcribe_and_diarize).
WHISPER_MODEL = "large-v3"
LANGUAGE = "en"

# 3.1 rather than community-1: community-1 is a separate gated repo and this
# token is only approved for the 3.1 pipeline + segmentation-3.0 behind it.
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Speaker count is left to pyannote by default. Set these when you know the
# recording (this case is two people, cook + camera operator) -- an unbounded
# clustering on 34 minutes of kitchen noise tends to invent a third speaker
# out of the clatter.
MIN_SPEAKERS = None
MAX_SPEAKERS = None

# Agglomerative clustering cut-off for "same voice". Left at pyannote's shipped
# value after trying to tune it: on this recording there is, as far as the
# microphone is concerned, one speaker.
#
# It is a head-mounted Aria: the wearer is loud and on-axis for 34 minutes,
# everyone else is quiet and off-axis, and the two channels are 0.9998
# correlated, so there is no spatial cue to separate anyone with either. At
# 0.7045 the whole file clusters as one speaker (903 s against 2.1 s of
# anything else). Lowering it does produce more clusters -- 3 at 0.55, 5 at
# 0.45, 8 at 0.40 -- but reading the transcript back, those clusters are the
# narrator fragmenting, not a second person appearing: "I'm gonna wash the
# knife while we wait" and "the stir-fry sauce looks very combined" both landed
# in the supposedly-other speaker. False turn boundaries in the middle of one
# person's sentences are worse than no diarization, so the default stands.
#
# What actually limits this is the VAD, not the clustering: the second person's
# quiet replies in the opening exchange are mostly below the speech threshold
# and never reach Whisper, so there are no words to attribute to them.
CLUSTERING_THRESHOLD = 0.7045

SAMPLE_RATE = 16000

# Silero's shipped 0.5 speech probability is tuned for a close, clean mic.
# Off-axis speech in this kitchen sits under it and is thrown away before
# Whisper sees it -- at 0.5 the first minute lost 27 of 101 words, including
# whole turns of the opening exchange. 0.2 keeps them. It also admits pan
# clatter as "speech", which is what NO_SPEECH_MAX is for; below 0.15 the
# regions grow until they swallow the silences again.
VAD_THRESHOLD = 0.2

# Whisper's own estimate that a decoded region held no speech. A clatter
# region still gets decoded into something -- on the noodles file "Thank you."
# at 00:00 and "Ssssssss" at 00:48, both at 0.85+, against 0.23-0.6 for the
# real speech around them -- so this is the line between the two.
NO_SPEECH_MAX = 0.6

# When VAD claims speech in less than this share of the recording, it is not
# to be believed and the file is decoded whole instead. Silero is trained on
# reasonably close-mic speech; a stadium is wall-to-wall crowd noise with the
# referee buried in it, and it passed 23.5 s of 269 s (9%) on the World Cup
# clip -- Whisper never saw the other four minutes, and the transcript came
# back with 31 words against Gemini's forty-odd exchanges. For contrast the
# recordings VAD handles properly sit far above this line: 51% on the noodles
# file, 73% on hide-and-seek.
VAD_FALLBACK_FRACTION = 0.25

# Audio tagging for the things Whisper cannot write down. Silero VAD only
# fires on speech, so a shriek or an animal impression never becomes a region
# and never reaches Whisper -- on the hide-and-seek clip the hiders' noises
# were absent from the transcript entirely while the seeker's commentary on
# them ("there's a tiger or something over here") was not. AudioSet's 527
# classes cover them.
EVENT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

# AST was trained on 10.24 s clips and its confidence falls off sharply on
# shorter ones -- the cattle impression at 01:43 scores 0.74 in a 5 s window
# and 0.23 in a 3 s one. 5 s with a 1 s hop keeps the scores while placing an
# event to within a second.
EVENT_WINDOW_S = 5.0
EVENT_HOP_S = 1.0
EVENT_MIN_SCORE = 0.4

# ...but a laugh is half a second long, and a 5 s window averages it away: the
# chuckle at 02:03 on the hide-and-seek clip scores 0.12 in a 1.5 s window and
# 0.03 in a 5 s one, below anything you would dare threshold on. Brief human
# vocalizations therefore get their own pass at a scale that fits them.
#
# The short window costs confidence across the board, so the threshold drops
# with it. That is affordable only because this pass is restricted to a
# handful of classes -- run over all 527 at 0.1 it would report constantly.
EVENT_VOCAL_WINDOW_S = 1.5
EVENT_VOCAL_HOP_S = 0.5
EVENT_VOCAL_MIN_SCORE = 0.10

# "Gasp" is deliberately absent, though it is the same kind of sound: it fired
# 22 times on the hide-and-seek clip and 15 on the noodles one, on breaths and
# wind rather than anything a reader would call a gasp. The classes here stay
# quiet unless something happens -- 3 hits and 0 respectively.
EVENT_VOCAL_LABELS = (
    "Laughter", "Giggle", "Chuckle, chortle", "Snicker", "Belly laugh",
    "Screaming", "Shout", "Yell", "Whoop", "Squeal", "Children shouting",
    "Cheering",
)

# Labels that describe the whole recording rather than a moment in it, plus
# the generic parents that always co-fire with the specific class underneath
# them ("Animal" alongside "Moo"). Keeping these would bury the events worth
# reading in a constant drone of "Speech" and "Outside, rural or natural".
EVENT_IGNORE = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Narration, monologue", "Conversation",
    "Speech synthesizer", "Babbling", "Silence",
    "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, rural or natural", "Outside, urban or manmade",
    "Animal", "Domestic animals, pets", "Livestock, farm animals, working animals",
    "Wild animals", "Vehicle", "Sound effect", "Music",
}

# A speaker run no longer than this, with the same speaker either side of it,
# is treated as a diarization boundary artefact rather than a turn.
MAX_ARTEFACT_WORDS = 3
MAX_ARTEFACT_S = 1.0

# A new utterance starts when the speaker changes, or when the same speaker
# leaves a gap this long -- otherwise a monologue comes back as one wall of
# text with a single timestamp on it, which is the thing that made the Gemini
# transcript unusable.
UTTERANCE_GAP_S = 2.0

# ...and a line holds at most this many seconds of speech, so a timecode
# always points at a span you can actually scrub to. Someone narrating
# steadily never leaves a 2 s gap, so without a limit the pauses do all the
# work and lines run to 30 s and beyond.
#
# It is a packing target, not a cut: lines are built out of whole sentences
# and a sentence is never broken to make one fit. A line therefore usually
# comes in under this, and a single long sentence is allowed to exceed it.
MAX_UTTERANCE_S = 10.0


# Subtrees of speechbrain.integrations that cannot be imported in this env:
# nlp needs flair, k2_fsa needs k2, numba needs numba. None of them is used --
# pyannote's speaker embeddings come from its own wespeaker wrapper.
_SPEECHBRAIN_UNUSABLE = (
    "speechbrain.integrations.nlp",
    "speechbrain.integrations.k2_fsa",
    "speechbrain.integrations.numba",
)


class _EmptyModuleFinder:
    """Import hook that hands back an empty module for the speechbrain
    integrations this environment cannot import, instead of letting them raise.

    speechbrain lazily exports every module under `integrations`, all the way
    down to leaves like numba.transducer_loss, and the import fires on
    *attribute access* rather than on use. So anything that walks module
    attributes sets one off -- autoreload's post-execute sweep, a debugger, a
    variable explorer, IPython's verbose traceback formatter. The last is the
    worst: tripped while rendering some other exception, its ImportError is
    what reaches the notebook and the real error is never seen.

    A finder rather than pre-registered stub modules in sys.modules, which is
    the obvious approach and the one tried first. A stub package has no
    __path__ into the real directory, so the submodules underneath it become
    unfindable and the crash simply moves down a level: stubbing
    integrations.numba turned the numba ImportError into a
    ModuleNotFoundError for integrations.numba.transducer_loss. A finder sits
    in front of the whole subtree at any depth and answers uniformly.

    Installed at the front of sys.meta_path but scoped to those three prefixes,
    so every other import in the process -- speechbrain's own, and everyone
    else's -- resolves normally."""

    def find_spec(self, fullname, path=None, target=None):
        if not any(fullname == p or fullname.startswith(p + ".")
                   for p in _SPEECHBRAIN_UNUSABLE):
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__doc__ = (
            f"Empty stand-in for {spec.name}, whose optional dependency is not "
            f"installed. Provided by {__name__} so that merely looking at this "
            f"attribute cannot raise."
        )
        module.__path__ = []  # a package, so children come back here too
        return module

    def exec_module(self, module):
        pass


def _defuse_speechbrain_lazy_integrations():
    """Install the finder once, and clear anything a previous version left in
    sys.modules so a kernel that already imported this module heals on reload
    rather than needing a restart."""
    if not any(isinstance(f, _EmptyModuleFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _EmptyModuleFinder())
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in _SPEECHBRAIN_UNUSABLE):
            del sys.modules[name]


_defuse_speechbrain_lazy_integrations()


def _hf_token():
    """pyannote 3.1 is gated, so the download needs the token. Read from the
    environment first, then the HF CLI's own login file -- `huggingface-cli
    login` is what put it there, and asking the user to also export it is a
    second place for it to go stale."""
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    stored = Path.home() / ".cache" / "huggingface" / "token"
    if stored.exists():
        return stored.read_text().strip()
    raise RuntimeError(
        "No Hugging Face token. Run `huggingface-cli login`, and accept the "
        f"conditions on https://huggingface.co/{DIARIZATION_MODEL} and "
        "https://huggingface.co/pyannote/segmentation-3.0."
    )


def _patch_hf_auth_kwarg():
    """pyannote 3.4 calls hf_hub_download(..., use_auth_token=...) unconditionally
    (core/pipeline.py and core/model.py), and huggingface_hub 1.x -- which the
    env's transformers 5.x requires -- deleted that argument in favour of
    `token`. The two are otherwise compatible, so the mismatch is renamed away
    here rather than pinning either side: pyannote 4.x, the version that already
    uses `token`, wants torch>=2.8 and would drag this env off the torch 2.6
    build the local SAM3 stage runs on.

    The modules import the function by name, so patching huggingface_hub alone
    would miss them -- each module's own reference is rebound too."""
    import huggingface_hub
    import pyannote.audio.core.model
    import pyannote.audio.core.pipeline

    original = huggingface_hub.hf_hub_download
    if getattr(original, "_auth_kwarg_patched", False):
        return

    def hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return original(*args, **kwargs)

    hf_hub_download._auth_kwarg_patched = True
    huggingface_hub.hf_hub_download = hf_hub_download
    pyannote.audio.core.pipeline.hf_hub_download = hf_hub_download
    pyannote.audio.core.model.hf_hub_download = hf_hub_download


def _allow_checkpoint_globals():
    """torch 2.6 flipped torch.load's `weights_only` default to True, and the
    pyannote checkpoints were pickled by an older Lightning that stores a
    TorchVersion object in the checkpoint metadata, plus pyannote's own task Specifications. Allow-listing those classes
    keeps the safe loader on, which is the point of the new default -- the
    blunt fix, weights_only=False, would disable the check for every checkpoint
    this process loads."""
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from torch.torch_version import TorchVersion
    torch.serialization.add_safe_globals(
        [TorchVersion, Specifications, Problem, Resolution])


def _preload_cuda_libs():
    """ctranslate2 (faster-whisper's backend) dlopen()s libcudnn/libcublas by
    soname, and this env has them only inside torch's pip-installed nvidia/*
    packages, which are not on the loader path. Editing LD_LIBRARY_PATH here
    would be too late -- the dynamic loader read it at process start -- so the
    libraries are opened by full path instead, which puts them in the process
    where ctranslate2's dlopen finds them already loaded.

    Failures are ignored on purpose: a system install of these libraries is
    also perfectly fine, and in that case there is nothing to preload."""
    import nvidia
    root = Path(nvidia.__file__).parent
    for pattern in ("cudnn/lib/libcudnn*.so.*", "cublas/lib/libcublas*.so.*"):
        for so in sorted(root.glob(pattern)):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


# All five are named for the case, never the experiment -- see the note on
# asr_transcript_path in A_Config. Re-running under a second experiment then
# finds the existing files and reuses them.
def audio_path():
    return asr_dir() / f"{case_name()}_audio16k.wav"


def words_path():
    return asr_dir() / f"{case_name()}_asr_words.json"


def rttm_path():
    return asr_dir() / f"{case_name()}_diarization.rttm"


def events_path():
    return asr_dir() / f"{case_name()}_audio_events.json"


def transcript_path():
    return asr_transcript_path()


def extract_audio(overwrite=False):
    """Video -> 16 kHz mono wav. Both models resample to 16k mono internally
    anyway; doing it once up front means the 2 GB mp4 is decoded once instead
    of twice, and the wav is small enough (~65 MB for 34 min) to keep."""
    out = audio_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        print(f"[asr] reusing {out.name}", flush=True)
        return out
    src = source_video_path()
    print(f"[asr] extracting audio from {src.name}", flush=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True,
    )
    return out


def speech_regions(audio):
    """Silero VAD regions, used exactly as VAD draws them.

    Joining regions across short pauses is tempting -- it reads better, since a
    knife stroke mid-sentence stops splitting the sentence -- but it was
    measured worse. Merged into 30 s chunks, Whisper locks onto the loud
    narrator and drops the quiet interjections around him: the opening
    exchange came back as one unbroken block with the other person's replies
    missing, and the punctuation degraded into a run-on. Short regions decoded
    on their own keep both.

    Capped at 30 s -- one Whisper window -- so no region needs internal
    stitching."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return get_speech_timestamps(audio, VadOptions(
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        max_speech_duration_s=30,
    ))


def run_whisper(overwrite=False):
    """Word-level transcription, one VAD region at a time.

    Not `model.transcribe(wav, vad_filter=True)`, which is the obvious way to
    write this and is what the first version did. That path cuts the silence
    out, decodes the surviving audio as one continuous stream, and maps the
    times back afterwards -- so a Whisper segment can span a removed silence,
    and its words get spread across a gap where nobody was talking. Measured
    here: "So, I'll" was stamped at 00:47-00:50 and the rest of the same
    sentence, "be doing the narration for the garlic noodles", at 03:10, with
    two and a half minutes of washing-up in between. The stitched context also
    corrupted the text itself -- "garlic" was decoded as "gloves." (p=0.13).
    Decoding each region separately gets both right, because the model only
    ever sees contiguous audio.

    Returns the list of word dicts (no speakers yet) and caches it as JSON so
    diarization can be re-run/re-tuned without paying for the ASR again."""
    out = words_path()
    if out.exists() and not overwrite:
        print(f"[asr] reusing {out.name}", flush=True)
        return json.loads(out.read_text())["words"]

    _preload_cuda_libs()
    import torchaudio
    from faster_whisper import WhisperModel

    wav = extract_audio()
    waveform, sr = torchaudio.load(str(wav))
    audio = waveform[0].numpy()
    regions = speech_regions(audio)
    speech_s = sum(r["end"] - r["start"] for r in regions) / sr
    duration_s = len(audio) / sr
    print(f"[asr] {len(regions)} speech regions, {speech_s / 60:.1f} min of speech",
          flush=True)

    if speech_s < VAD_FALLBACK_FRACTION * duration_s:
        print(f"[asr] VAD found speech in only {speech_s / duration_s:.0%} of the "
              f"recording -- too noisy to gate on, decoding it whole", flush=True)
        regions = [{"start": 0, "end": len(audio)}]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(
        WHISPER_MODEL, device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )
    print(f"[asr] whisper {WHISPER_MODEL} on {device}", flush=True)

    words, dropped = [], 0
    next_report = 0
    for region in regions:
        offset = region["start"] / sr
        segments, _ = model.transcribe(
            audio[region["start"]:region["end"]],
            language=LANGUAGE,
            word_timestamps=True,
            # The regions are already the speech; a second VAD pass inside
            # them would only re-introduce the stitching this loop avoids.
            vad_filter=False,
            # Nothing carries between regions -- they are minutes apart in
            # places, and priming a region with the last one's text is how
            # Whisper gets into a repetition loop.
            condition_on_previous_text=False,
        )
        for seg in segments:
            # A region that turns out to be clatter rather than speech still
            # gets decoded into something, usually a stock phrase ("Thank
            # you.", "Bye."). Whisper's own no-speech estimate is the cheapest
            # filter for those, and it is only consulted for regions VAD
            # already thought were marginal.
            if seg.no_speech_prob > NO_SPEECH_MAX:
                dropped += len(seg.words or [])
                continue
            for w in (seg.words or []):
                words.append({
                    "start": round(w.start + offset, 3),
                    "end": round(w.end + offset, 3),
                    "word": w.word,
                    "prob": round(w.probability, 3),
                })
        if offset >= next_report:
            print(f"[asr]   ...{int(offset) // 60:02d}:{int(offset) % 60:02d}", flush=True)
            next_report = offset + 300

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": WHISPER_MODEL,
        "language": LANGUAGE,
        "duration_s": round(len(audio) / sr, 2),
        "speech_s": round(speech_s, 2),
        "words": words,
    }, indent=1))
    print(f"[asr] {len(words)} words ({dropped} dropped as non-speech) -> {out.name}",
          flush=True)

    del model
    torch.cuda.empty_cache()
    return words


def run_diarization(overwrite=False):
    """pyannote turns as (start_s, end_s, speaker) tuples, also written as RTTM
    for anything that wants the standard format."""
    out = rttm_path()
    if out.exists() and not overwrite:
        print(f"[asr] reusing {out.name}", flush=True)
        return _read_rttm(out)

    from pyannote.audio import Pipeline

    _patch_hf_auth_kwarg()
    _allow_checkpoint_globals()

    wav = extract_audio()
    # pyannote 3.x names this use_auth_token; 4.x renamed it to token.
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=_hf_token())
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    print(f"[asr] diarizing with {DIARIZATION_MODEL}", flush=True)

    pipeline.clustering.threshold = CLUSTERING_THRESHOLD

    kwargs = {}
    if MIN_SPEAKERS is not None:
        kwargs["min_speakers"] = MIN_SPEAKERS
    if MAX_SPEAKERS is not None:
        kwargs["max_speakers"] = MAX_SPEAKERS
    annotation = pipeline(str(wav), **kwargs)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        annotation.write_rttm(f)
    turns = [(seg.start, seg.end, spk)
             for seg, _, spk in annotation.itertracks(yield_label=True)]
    print(f"[asr] {len(turns)} turns, {len(set(t[2] for t in turns))} speakers", flush=True)

    del pipeline
    torch.cuda.empty_cache()
    return turns


def run_audio_events(overwrite=False):
    """Non-speech audio events, as {start, end, label, score} spans.

    Deliberately independent of the VAD regions: the whole point is to catch
    what VAD rejected. The tagger sees a sliding window over the entire file.

    Two passes at two scales, because sounds have durations: a five-second
    window for sustained activity (a running tap, chopping, a pan sizzling)
    and a 1.5-second one for brief human vocalizations, which the long window
    averages into nothing.

    What it finds is not always what you would name yourself. On the
    hide-and-seek clip the children's silly noises come back as "Cattle,
    bovinae" (0.74) and "Roar" (0.58) -- which is right, they are doing animal
    impressions. Read the labels as what the sound resembles, not as a
    transcript of it."""
    out = events_path()
    if out.exists() and not overwrite:
        print(f"[asr] reusing {out.name}", flush=True)
        return json.loads(out.read_text())["events"]

    import numpy as np
    import torchaudio
    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    wav = extract_audio()
    waveform, sr = torchaudio.load(str(wav))
    audio = waveform[0].numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(EVENT_MODEL)
    model = ASTForAudioClassification.from_pretrained(EVENT_MODEL).to(device).eval()
    id2label = model.config.id2label
    print(f"[asr] tagging audio events with {EVENT_MODEL.split('/')[-1]}", flush=True)

    def scan(window_s, hop_s, min_score, keep=None):
        """Slide `window_s` over the whole file and return the surviving spans.

        `keep`, when given, restricts scoring to those labels; otherwise the
        top-scoring label that isn't context or a generic parent wins the
        window."""
        starts = np.arange(0.0, max(len(audio) / sr - window_s, 0.0) + hop_s, hop_s)
        per_window = []
        batch = 16
        for i in range(0, len(starts), batch):
            chunk = [audio[int(t * sr):int((t + window_s) * sr)]
                     for t in starts[i:i + batch]]
            inputs = extractor(chunk, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                # Multi-label head: sigmoid per class, not a softmax over
                # classes -- several things can be audible at once.
                probs = torch.sigmoid(model(**inputs).logits).cpu().numpy()
            for t, row in zip(starts[i:i + batch], probs):
                best, best_score = None, 0.0
                for j in row.argsort()[::-1][:10]:
                    label = id2label[int(j)]
                    if keep is not None:
                        if label not in keep:
                            continue
                    elif label in EVENT_IGNORE:
                        continue
                    best, best_score = label, float(row[j])
                    break
                if best is not None and best_score >= min_score:
                    per_window.append((t, best, best_score))

        # Neighbouring windows overlap, so one real event shows up in several
        # of them; collapse a consecutive run of the same label into the span
        # it covers and keep its strongest score.
        spans = []
        for t, label, score in per_window:
            cur = spans[-1] if spans else None
            if cur and cur["label"] == label and t - cur["end"] <= window_s:
                cur["end"] = round(t + window_s, 2)
                cur["score"] = max(cur["score"], round(score, 3))
            else:
                spans.append({"start": round(t, 2),
                              "end": round(t + window_s, 2),
                              "label": label,
                              "score": round(score, 3)})

        # A continuous activity drifts between neighbouring AudioSet classes as
        # it goes -- 37 seconds at the sink came back as five spans alternating
        # "Water", "Water tap, faucet" and "Sink (filling or washing)".
        # Overlapping spans are the same moment described more than once, so
        # they collapse into one carrying whichever label scored highest.
        collapsed = []
        for span in spans:
            cur = collapsed[-1] if collapsed else None
            if cur and span["start"] < cur["end"]:
                cur["end"] = max(cur["end"], span["end"])
                if span["score"] > cur["score"]:
                    cur["label"], cur["score"] = span["label"], span["score"]
            else:
                collapsed.append(span)
        return collapsed

    # The two passes are kept apart rather than collapsed together: a laugh
    # scoring 0.12 inside a five-second "Sizzle" at 0.70 is not the same moment
    # described twice, and merging by score would silently drop it.
    events = scan(EVENT_WINDOW_S, EVENT_HOP_S, EVENT_MIN_SCORE)
    vocal = scan(EVENT_VOCAL_WINDOW_S, EVENT_VOCAL_HOP_S, EVENT_VOCAL_MIN_SCORE,
                 keep=set(EVENT_VOCAL_LABELS))
    events = sorted(events + vocal, key=lambda e: (e["start"], e["end"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": EVENT_MODEL,
        "window_s": EVENT_WINDOW_S,
        "hop_s": EVENT_HOP_S,
        "min_score": EVENT_MIN_SCORE,
        "vocal_window_s": EVENT_VOCAL_WINDOW_S,
        "vocal_hop_s": EVENT_VOCAL_HOP_S,
        "vocal_min_score": EVENT_VOCAL_MIN_SCORE,
        "events": events,
    }, indent=1))
    print(f"[asr] {len(events)} audio events ({len(vocal)} vocal) -> {out.name}",
          flush=True)

    del model
    torch.cuda.empty_cache()
    return events


def _read_rttm(path):
    turns = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "SPEAKER":
            start, dur = float(parts[3]), float(parts[4])
            turns.append((start, start + dur, parts[7]))
    return turns


def _speaker_for(word, turns):
    """The speaker whose turn overlaps this word most. Whisper's word bounds
    and pyannote's turn bounds are drawn by different models and rarely line
    up exactly, so a containment test would leave every boundary word
    unlabelled; maximum overlap always yields an answer where there is any
    overlap at all. A word inside a gap in the diarization (breath, clatter
    mislabelled as speech) genuinely has no speaker and gets None."""
    best, best_overlap = None, 0.0
    for start, end, spk in turns:
        overlap = min(word["end"], end) - max(word["start"], start)
        if overlap > best_overlap:
            best, best_overlap = spk, overlap
    return best


def _mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def _smooth_speakers(labelled):
    """Relabel a brief speaker run that interrupts one continuous speaker.

    Diarization boundaries land a word or two out, so a run of narration comes
    back with the odd word attributed to a minor cluster -- "**03:25-03:26**
    [SPEAKER_01]: the" between two SPEAKER_02 sentences, which then splits the
    sentence into three lines. A run that is this short and has the *same*
    speaker either side of it is a boundary artefact, not a turn: nobody says
    one word and hands the floor straight back.

    Only flanked runs are touched. A short run between two different speakers
    is exactly what a real backchannel looks like ("Yeah." while the other
    person carries on), so those are left alone."""
    runs = []
    for w in labelled:
        if runs and runs[-1][0] == w["speaker"]:
            runs[-1][1].append(w)
        else:
            runs.append([w["speaker"], [w]])
    for i in range(1, len(runs) - 1):
        spk, run = runs[i]
        brief = (len(run) <= MAX_ARTEFACT_WORDS
                 and run[-1]["end"] - run[0]["start"] <= MAX_ARTEFACT_S)
        if brief and runs[i - 1][0] == runs[i + 1][0] and spk != runs[i - 1][0]:
            for w in run:
                w["speaker"] = runs[i - 1][0]
    return labelled


def _ends_sentence(word):
    return word["word"].rstrip().endswith((".", "?", "!"))


def _sentences(block):
    """A block of words -> one list per sentence, split after its full stop.

    Whisper punctuates well when it is confident and not at all when it isn't
    -- whole regions of this material come back as unbroken lowercase -- so a
    trailing run with no full stop in it is returned as it stands and dealt
    with by _split_long."""
    out, cur = [], []
    for w in block:
        cur.append(w)
        if _ends_sentence(w):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _split_long(sentence):
    """Break an over-long unpunctuated run at its longest internal silence.

    Only reached where Whisper gave no punctuation to break on. The longest
    pause is the best guess available at where the speaker finished a thought,
    and it beats cutting at a word count, which lands mid-phrase every time."""
    span = sentence[-1]["end"] - sentence[0]["start"]
    if span <= MAX_UTTERANCE_S or len(sentence) < 4:
        return [sentence]
    gaps = [(sentence[i + 1]["start"] - sentence[i]["end"], i)
            for i in range(len(sentence) - 1)]
    _, at = max(gaps)
    return _split_long(sentence[:at + 1]) + _split_long(sentence[at + 1:])


def merge(words, turns):
    """Words + turns -> utterances: whole sentences, packed up to
    MAX_UTTERANCE_S, split on speaker changes and pauses.

    Sentences are the unit rather than seconds. Packing by time alone gives
    every line the same width and none of them an ending -- "...so how we use
    it is so we" / "and we want to cook 8 oz of dry noodles so okay so before"
    -- which is tidy to look at and unreadable. So the block is cut into
    sentences first and lines are filled with as many whole sentences as fit;
    a sentence longer than the target simply gets a longer line.

    An unlabelled word inherits the utterance it lands in rather than breaking
    it, so a single missed word mid-sentence doesn't fragment the line."""
    labelled = _smooth_speakers([
        dict(w, speaker=_speaker_for(w, turns) or "UNKNOWN") for w in words
    ])

    # Blocks first: a speaker change or a real pause ends a line no matter
    # where the sentence had got to.
    blocks = []
    for w in labelled:
        cur = blocks[-1] if blocks else None
        if (cur and w["speaker"] == cur[-1]["speaker"]
                and w["start"] - cur[-1]["end"] <= UTTERANCE_GAP_S):
            cur.append(w)
        else:
            blocks.append([w])

    utterances = []
    for block in blocks:
        for sentence in _sentences(block):
            for piece in _split_long(sentence):
                cur = utterances[-1] if utterances else None
                fits = (cur is not None
                        and cur["speaker"] == piece[0]["speaker"]
                        and cur["block"] is block
                        and piece[-1]["end"] - cur["start"] <= MAX_UTTERANCE_S)
                if fits:
                    cur["words"].extend(piece)
                    cur["end"] = piece[-1]["end"]
                else:
                    utterances.append({
                        "speaker": piece[0]["speaker"],
                        "start": piece[0]["start"],
                        "end": piece[-1]["end"],
                        "words": list(piece),
                        "block": block,
                    })
    for u in utterances:
        u["text"] = " ".join(x["word"].strip() for x in u["words"]).strip()
        del u["block"]
    return utterances


def write_transcript(utterances, events=()):
    """Same shape as the Gemini transcript so downstream readers need no
    change, with an end time added -- **MM:SS-MM:SS** [SPEAKER_00]: text.

    Audio events are interleaved in time on a [sound] pseudo-speaker, so the
    one file reads as an account of the recording rather than of its speech
    alone. They sort by start time alongside the utterances; where an event
    overlaps speech it lands next to the line it happened during."""
    out = transcript_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [(u["start"], u["end"], u["speaker"], u["text"]) for u in utterances]
    rows += [(e["start"], e["end"], "sound", f"{e['label']} ({e['score']:.2f})")
             for e in events]
    lines = [f"**{_mmss(start)}-{_mmss(end)}** [{who}]: {text}"
             for start, end, who, text in sorted(rows)]
    out.write_text("\n\n".join(lines) + "\n")
    print(f"[asr] {len(utterances)} utterances + {len(events)} events -> {out}",
          flush=True)
    return out


def transcribe_and_diarize(overwrite=False):
    """The whole stage: what was said, who said it, and what else was audible.

    The three models run one after the other, never together -- large-v3 in
    fp16 plus the diarization pipeline both resident would be tight on 24 GB,
    and there is nothing to gain from overlapping them since each saturates
    the card on its own."""
    words = run_whisper(overwrite=overwrite)
    turns = run_diarization(overwrite=overwrite)
    events = run_audio_events(overwrite=overwrite)
    utterances = merge(words, turns)

    # Speaker-labelled words are the useful machine-readable artefact (the txt
    # is for humans), so fold the labels back into the cached JSON.
    # These are merge()'s own word dicts, not the list run_whisper returned:
    # smoothing relabels copies, so writing the originals back saved a
    # `speaker` of None against every word while the transcript beside it
    # showed labels.
    labelled = [w for u in utterances for w in u["words"]]
    payload = json.loads(words_path().read_text())
    payload["words"] = labelled
    payload["speakers"] = sorted({u["speaker"] for u in utterances})
    words_path().write_text(json.dumps(payload, indent=1))

    return write_transcript(utterances, events)


if __name__ == "__main__":
    import sys
    from A_Config import set_case
    set_case(sys.argv[1])
    transcribe_and_diarize()
