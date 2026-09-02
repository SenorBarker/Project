
from A_Config import set_case, case_dir
import json
from pathlib import Path
from B_video_processing import frame_times_at_indices
from A_Config import set_case, source_video_path, source_dir, REPO_ROOT, agent_p_output_dir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter


BIN_S = 1.0
# One knob for every bit of text in every figure in this module: titles, axis
# labels, tick labels, legends. Set it before plotting, or call set_font_size().
FONT_SIZE = 12
plt.rcParams.update({"font.size": FONT_SIZE})


def set_font_size(size):
    """Resize every bit of text in this module's figures."""
    global FONT_SIZE
    FONT_SIZE = size
    plt.rcParams.update({"font.size": size})
project_root = Path.cwd()
#----------VS ME-------------ALIVE
from B_video_processing import video_duration_sec

def time_grid(case_name, bin_s=BIN_S, labels=None, tol_s=1.0):
    """Shared time axis for a case. Prefers the video's own duration; falls back
    to the annotators' stated duration when the footage isn't on this machine."""
    set_case(case_name)
    vids = list(source_dir().glob("*.mp4"))
    stated = None
    if labels:
        d = {round(l["video_duration_sec"], 2) for l in labels}
        assert len(d) == 1, f"{case_name}: annotators disagree on duration: {d}"
        stated = d.pop()

    if vids:
        dur = video_duration_sec(vids[0])
        if stated is not None:
            assert abs(stated - dur) <= tol_s, (
                f"{case_name}: labels say {stated}s, video is {dur:.2f}s — wrong pairing?")
    else:
        assert stated is not None, f"{case_name}: no video and no labels to get duration from"
        print(f"{case_name}: no source video — grid from labels ({stated}s)")
        dur = stated

    n = int(np.ceil(dur / bin_s))
    edges = np.arange(n + 1) * bin_s
    return edges

def spans_to_mask(spans, edges):
    """(start_sec, end_sec) pairs -> kept/not mask on the grid."""
    n = len(edges) - 1
    bin_s = edges[1] - edges[0]
    included = np.zeros(n, dtype=bool)
    for s, e in spans:
        i0 = max(0, int(np.floor(s / bin_s)))
        i1 = min(n, max(i0 + 1, int(np.ceil(e / bin_s))))
        included[i0:i1] = True
    return included

def cut_csv_to_timegrid(csv_path, edges):
    """My reference cut -> per-second kept/not, on the vote grid."""
    df = pd.read_csv(csv_path)
    spans = list(zip(df.src_in_sec, df.src_out_sec))
    return spans_to_mask(spans, edges)


def GT_cut_path(case_name):
    """Always Data/<case>/100_labels/<case>_cuts.csv -- fixed convention,
    no per-case exceptions."""
    set_case(case_name)
    return case_dir() / "100_labels" / f"{case_name}_cuts.csv"


def resolve_GT_edit(case_name, edges):
    """This case's reference cut as a mask, or None if it hasn't been cut yet --
    a lookup by case_name alone, no path to type out per case."""
    path = GT_cut_path(case_name)
    if not path.exists():
        return None
    return cut_csv_to_timegrid(path, edges)


def paper_edit_to_timegrid(paper_edit_json_path, case_name, edges):
    """Paper edit -> per-second kept/not, on the same grid as the vote curve.
    Beat bounds are frame indices, converted via each frame's own decoded
    timestamp (no fps assumption). Unresolved beats are skipped and reported."""
    beats = json.loads(Path(paper_edit_json_path).read_text())["beats"]

    frame_spans = [(b["start_frame"], b["end_frame"]) for b in beats
                   if b.get("start_frame") is not None and b.get("end_frame") is not None]
    skipped = len(beats) - len(frame_spans)
    if skipped:
        print(f"{case_name}: {skipped}/{len(beats)} beats have no resolved frames — skipped")

    set_case(case_name)
    times = frame_times_at_indices(source_video_path(),
                                   {f for span in frame_spans for f in span})  # ONE pass

    spans = [(times[fin], times[fout]) for fin, fout in frame_spans
             if fin in times and fout in times]              # frames -> seconds
    dropped = len(frame_spans) - len(spans)
    if dropped:
        print(f"{case_name}: {dropped} beats outside video — skipped")

    return spans_to_mask(spans, edges)


def paper_edit_path(case_name, experiment):
    """Always Data/<case>/012_agent_p_output/<experiment>/<case>_<experiment>_1_revision.json.
    Not guaranteed for every run (drafts, feedback passes etc. use other
    suffixes) but this is the convention the eval notebooks target."""
    set_case(case_name, experiment)
    return agent_p_output_dir() / f"{case_name}_{experiment}_1_revision.json"


def resolve_paper_edit(case_name, experiment, edges):
    """The machine's cut for case_name/experiment as a mask, or None if that
    run hasn't produced a revision yet -- a lookup by (case, experiment),
    no path to type out per case."""
    path = paper_edit_path(case_name, experiment)
    if not path.exists():
        return None
    return paper_edit_to_timegrid(path, case_name, edges)



_CRITICAL_REGISTRY = {}  # id(full mask) -> that annotator's critical-only mask,
# so timeline_agreement can auto-hatch critical beats without the caller having
# to build or pass a second track for it.


def critical_union(labels, edges):
    """Every second any annotator flagged critical, overlaps merged so a second
    two annotators both marked critical is counted once. This is the denominator
    for "did we catch the critical moments" -- how much critical material exists
    at all, not how many critical beats were written."""
    n = len(edges) - 1
    union = np.zeros(n, dtype=bool)
    for l in labels:
        union |= annotator_beats_to_grid(l, edges, event_category="critical")
    return union


def critical_beats_hit(labels, edges, **edits):
    """Whole critical BEATS hit, not seconds. Denominator is every critical beat
    any annotator wrote, one count per beat -- two annotators each flagging
    their own overlapping beat critical counts as 2 votes here, unlike
    critical_union which would merge that overlap into one span. This is 'how
    many of the critical calls did each editor's cut land inside', not 'how
    much critical runtime did it cover'."""
    bin_s = edges[1] - edges[0]
    n = len(edges) - 1
    crit_beats = [(b["start_sec"], b["end_sec"]) for l in labels for b in l["beats"]
                 if b.get("event_category") == "critical"]
    total = len(crit_beats)

    rows = []
    for name, mask in edits.items():
        hit = 0
        for s, e in crit_beats:
            i0 = max(0, int(np.floor(s / bin_s)))
            i1 = min(n, max(i0 + 1, int(np.ceil(e / bin_s))))
            if mask[i0:i1].any():          # any overlap counts as a hit
                hit += 1
        rows.append({
            "editor": name,
            "Critical beats hit": hit,
            "Critical beats voted": total,
            "Crit beat rate": hit / total if total else np.nan,
        })
    return pd.DataFrame(rows)


def critical_beats_hit_all_cases(case_experiments):
    """critical_beats_hit summed across every case into one row per editor
    (GT, machine) -- total critical beats hit vs total voted, not broken out
    per case. Cases missing a GT cut or that experiment's revision are skipped
    and reported, exactly as critical_hit_tally_all_cases does."""
    hits, totals = {}, {}
    for case, experiment in case_experiments.items():
        set_case(case)
        try:
            labels = load_labels()
        except FileNotFoundError as e:
            print(f"{case}: skipped, {e}")
            continue
        edges = time_grid(case, labels=labels)

        GT = resolve_GT_edit(case, edges)
        if GT is None:
            print(f"{case}: skipped, no GT cut at {GT_cut_path(case)}")
            continue
        machine = resolve_paper_edit(case, experiment, edges)
        if machine is None:
            print(f"{case}/{experiment}: skipped, no revision at {paper_edit_path(case, experiment)}")
            continue

        df = critical_beats_hit(labels, edges, GT=GT, machine=machine)
        voted = int(df["Critical beats voted"].iloc[0])
        gt_hit = int(df.loc[df.editor == "GT", "Critical beats hit"].iloc[0])
        m_hit = int(df.loc[df.editor == "machine", "Critical beats hit"].iloc[0])
        print(f"{case:24s} exp={experiment:16s} voted={voted:3d} GT={gt_hit:3d} machine={m_hit:3d}")
        for _, row in df.iterrows():
            hits[row["editor"]] = hits.get(row["editor"], 0) + row["Critical beats hit"]
            totals[row["editor"]] = totals.get(row["editor"], 0) + row["Critical beats voted"]

    return pd.DataFrame([
        {"editor": e, "Critical beats hit": hits[e], "Critical beats voted": totals[e],
         "Crit beat rate": hits[e] / totals[e] if totals[e] else np.nan}
        for e in hits
    ])


def critical_hit_tally(labels, edges, **edits):
    """Coverage of the critical union by each of `edits` (name=mask, e.g.
    GT_edit=GT_edit, machine=machine_paper_edit). Returns a DataFrame:
    "Critical hits (s)" seconds of that editor's cut inside the critical union,
    "Vote critical (s)" the total critical seconds any annotator flagged (same
    for every row -- the denominator), "Crit rate" the fraction covered."""
    crit = critical_union(labels, edges)
    total_s = float(crit.sum() * BIN_S)
    rows = []
    for name, mask in edits.items():
        hit_s = float((mask & crit).sum() * BIN_S)
        rows.append({
            "editor": name,
            "Critical hits (s)": hit_s,
            "Vote critical (s)": total_s,
            "Crit rate": hit_s / total_s if total_s else np.nan,
        })
    return pd.DataFrame(rows)


def critical_hit_tally_all_cases(case_experiments):
    """critical_hit_tally run across every case, stacked into one table.

    case_experiments: {case_name: experiment} -- e.g. {"402_Hide_n_seek":
    "Pipe_eval_01"}. Both the GT cut and the machine's paper edit are looked
    up by (case, experiment) alone via resolve_GT_edit / resolve_paper_edit --
    nothing to load or path out by hand. A case missing its GT cut or that
    experiment's revision on disk is skipped and reported, not a crash."""
    rows = []
    for case, experiment in case_experiments.items():
        set_case(case)
        try:
            labels = load_labels()
        except FileNotFoundError as e:
            print(f"{case}: skipped, {e}")
            continue
        edges = time_grid(case, labels=labels)

        GT = resolve_GT_edit(case, edges)
        if GT is None:
            print(f"{case}: skipped, no GT cut at {GT_cut_path(case)}")
            continue
        machine = resolve_paper_edit(case, experiment, edges)
        if machine is None:
            print(f"{case}/{experiment}: skipped, no revision at {paper_edit_path(case, experiment)}")
            continue

        df = critical_hit_tally(labels, edges, GT=GT, machine=machine)
        df.insert(0, "experiment", experiment)
        df.insert(0, "case", case)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def annotator_beats_to_grid(label, edges, event_category=None):
    """One annotator's own selection, as a kept/not mask.
    Pass event_category="critical" (etc.) to keep only beats of that type.
    On an unfiltered call, also registers that annotator's critical-only mask
    against this mask's id() -- timeline_agreement picks it up automatically."""
    spans = [(b["start_sec"], b["end_sec"]) for b in label["beats"]
             if event_category is None or b.get("event_category") == event_category]
    mask = spans_to_mask(spans, edges)
    if event_category is None:
        crit_spans = [(b["start_sec"], b["end_sec"]) for b in label["beats"]
                     if b.get("event_category") == "critical"]
        crit_mask = spans_to_mask(crit_spans, edges)
        if crit_mask.any():
            _CRITICAL_REGISTRY[id(mask)] = crit_mask
    return mask



def proximity_credit(mask, decay_s=4.0):
    """1.0 inside the mask, falling linearly to 0 at decay_s seconds away."""
    steps = max(1, int(round(decay_s / BIN_S)))
    credit = mask.astype(float)
    cur = mask.copy()
    for d in range(1, steps):
        cur = _dilate1(cur)
        credit = np.maximum(credit, (1 - d / steps) * cur)
    return credit



def score_paper_vs_GT_edits(GT_edit, paper_edit, beta=1.0, decay_s=4.0):
    """Grade one paper edit against my reference cut.

    Both args are boolean masks on the same grid. TN is deliberately not scored —
    correct exclusions outnumber everything else and would swamp the result.

    Hard metrics treat a second as in or out. Soft metrics give partial credit by
    proximity: a kept second 1s outside my cut scores 0.75, 2s 0.5, 3s 0.25, and
    4s+ nothing — so sloppy boundaries barely register but isolated junk is fully
    punished. Symmetric: a GT second just outside their cut counts as nearly
    covered rather than a full miss."""
    assert GT_edit.shape == paper_edit.shape, "masks on different grids"

    tp = float((paper_edit & GT_edit).sum())
    fp = float((paper_edit & ~GT_edit).sum())
    fn = float((~paper_edit & GT_edit).sum())
    tn = float((~paper_edit & ~GT_edit).sum())

    # correct vs total in the paper edit
    precision = tp / (tp + fp) if (tp + fp) else np.nan     # did it trim like me
    # correct vs total in the GT
    recall    = tp / (tp + fn) if (tp + fn) else np.nan     # did it find my material
    #correct omissions (paper) vs all my omissions
    specificity = tn /(tn + fp) if (tn + fp)else np.nan

    # correct vs all seconds in ANY cut
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    #preciusion combined with recall
    f_beta    = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)
                 if precision and recall else 0.0)

    # boundary-tolerant versions
    soft_gt = proximity_credit(GT_edit, decay_s)
    soft_paper_edit = proximity_credit(paper_edit, decay_s)
    soft_precision = float(soft_gt[paper_edit].mean()) if paper_edit.any() else np.nan
    soft_recall    = float(soft_paper_edit[GT_edit].mean())  if GT_edit.any() else np.nan
    #is it in the paper edit, if yes, get the GT value (0 is neg, frac if near + 1 if +), 
    # if no get 1 (Tn). Then restrict to just the bits NOT in the GT (which is TN and FP)
    soft_specificity =float(np.where(paper_edit, soft_gt,1.0)[~GT_edit].mean())
    soft_f1 = (2 * soft_precision * soft_recall / (soft_precision + soft_recall)
               if soft_precision and soft_recall else 0.0)

    # fragmentation — reported, not scored; use it as a gate
    d = np.diff(np.concatenate(([0], paper_edit.view(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    seg_lens = (ends - starts) * BIN_S

    return dict(
        
        soft_precision = soft_precision,
        precision      = precision,
        soft_recall    = soft_recall,
        recall         = recall,
        soft_specificity =soft_specificity,
        specificity = specificity,
        iou            = iou,    
        soft_f1        = soft_f1,      
        f_beta         = f_beta,
        beta           = beta,
        decay_s        = decay_s,
        true_p_s           = tp * BIN_S,
        false_p_s           = fp * BIN_S,
        false_n_s           = fn * BIN_S,
        true_n_s           = tn *BIN_S,
        edit_s         = float(paper_edit.sum() * BIN_S),
        gt_s           = float(GT_edit.sum() * BIN_S),
        len_ratio      = float(paper_edit.sum() / GT_edit.sum()) if GT_edit.any() else np.nan,
        n_segments     = len(seg_lens),
        shortest_seg_s = float(seg_lens.min()) if len(seg_lens) else np.nan,
        median_seg_s   = float(np.median(seg_lens)) if len(seg_lens) else np.nan,
    )

#------RANDOM
def random_baseline_closed_form(GT_edit, paper_edit):
    """Uniformly random second-picking. Exact, hard measures only."""
    N = float(len(GT_edit))
    g, e = float(GT_edit.sum()), float(paper_edit.sum())
    tp = e * g / N
    return dict(
        editor      = "random (uniform)",
        precision   = g / N,
        recall      = e / N,
        iou         = tp / (g + e - tp),
        f_beta      = 2 * e * g / (N * (e + g)),
        specificity = 1 - (e - tp) / (N - g),
        edit_s      = e * BIN_S,
        gt_s        = g * BIN_S,
        len_ratio   = e / g,
    )


def random_baseline_blocks(GT_edit, paper_edit, n_draws=200, seed=0,
                           report=("iou", "soft_f1", "precision", "recall", "specificity")):
    """Null that keeps the editor's own segment count and lengths, placed at random.
    Returns (row, stats, density): the mean baseline row; for each reported
    measure, the 5th/95th percentile of the null and where the real edit sits
    inside it; and density, the per-second fraction of the 200 draws that kept
    that second — for plotting as a track alongside the real edits (it should
    come out as a near-uniform wash, unlike every real editor's hard bands)."""
    rng = np.random.default_rng(seed)
    n = len(paper_edit)
    d = np.diff(np.concatenate(([0], paper_edit.view(np.int8), [0])))
    lens = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)

    draws = []
    density = np.zeros(n)
    for _ in range(n_draws):
        m = np.zeros(n, dtype=bool)
        for L in sorted(rng.permutation(lens), reverse=True):   # longest first
            for _ in range(50):
                i = rng.integers(0, max(1, n - L))
                if not m[i:i + L].any():
                    m[i:i + L] = True
                    break
        density += m
        draws.append(score_paper_vs_GT_edits(GT_edit, m))
    density /= n_draws

    actual = score_paper_vs_GT_edits(GT_edit, paper_edit)
    num_keys = [k for k, v in draws[0].items() if isinstance(v, (int, float))]
    row = {k: float(np.nanmean([dr[k] for dr in draws])) for k in num_keys}
    row["editor"] = f"random (blocks, n={n_draws})"

    stats = {}
    for k in report:
        vals = np.array([dr[k] for dr in draws], dtype=float)
        stats[k] = dict(
            Producer_agent_score = actual[k],
            mean   = float(np.nanmean(vals)),
            p05    = float(np.nanpercentile(vals, 5)),
            p95    = float(np.nanpercentile(vals, 95)),
            pct    = float(np.mean(vals < actual[k]) * 100),   # where the real edit sits
        )
    return row, pd.DataFrame(stats).T.round(3), density
#--------------------------VIS
TP_C, FP_C, FN_C, REF_C, TN_C = "#1baf7a", "#e34948", "#c3c2b7", "#eb6834", "#fcfcfb"


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _row_image(mask, GT_edit=None, ref_color=None):
    """One row as an (1, n, 3) RGB array, one solid pixel per second -- no
    polygon edges between adjacent bins, so there is nothing for a rendering
    seam to appear between, unlike stacking several fill_between calls."""
    n = len(mask)
    img = np.empty((1, n, 3))
    if ref_color is not None:
        img[0, mask] = _hex_to_rgb(ref_color)
        img[0, ~mask] = _hex_to_rgb(TN_C)
    else:
        img[0, mask & GT_edit]  = _hex_to_rgb(TP_C)
        img[0, mask & ~GT_edit] = _hex_to_rgb(FP_C)
        img[0, ~mask & GT_edit] = _hex_to_rgb(FN_C)
        img[0, ~mask & ~GT_edit] = _hex_to_rgb(TN_C)
    return img

ANON = {"George": "A1", "Tamara": "A2", "Vuk": "A3", "stefanos": "A4",
        "ME (GT)": "Reference editor", "machine": "System"}



def timeline_agreement(case_name, edges, tracks, GT_edit=None, votes=None,
                       hatch=None, fname=None, xlim=None):
    """One band per editor on a shared time axis. If GT_edit is given, every
    boolean track except the reference is coloured by agreement with it: green
    = kept and in the reference, red = kept but not, grey = missed.
    Pass votes=<curve> to add the panel vote histogram as a top panel.
    A track may also be a float array in [0, 1] (e.g. random_baseline_blocks'
    density) -- drawn as a greyscale wash instead of a hard band, so a null with
    no real structure reads visually distinct from every real editor.
    hatch: optional {label: mask} to add or override which seconds get
    hatched. Every annotator track already hatches its own critical beats
    automatically (registered by annotator_beats_to_grid) -- nothing needs
    passing here for that; this is only for extra/custom hatches."""
    n = len(tracks)
    has_votes = votes is not None

    fig, axes = plt.subplots(n + has_votes, 1,
                             figsize=(14, 1.2 + 0.45 * n + 1.5 * has_votes),
                             sharex=True, squeeze=False,
                             height_ratios=([3] if has_votes else []) + [1] * n,
                             gridspec_kw=dict(hspace=0.15))
    axes = axes[:, 0]

    if has_votes:
        axes[0].stairs(votes, edges, fill=True, color="#2a78d6")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("P(include)")

    track_axes = axes[has_votes:]
    axes[0].set_title(case_name)

    t = edges[:-1]
    has_hatch = False
    for ax, (label, mask) in zip(track_axes, tracks.items()):
        if mask.dtype != bool:                                   # density row
            ax.imshow(mask[None, :], aspect="auto", cmap="Greys", vmin=0, vmax=1,
                      extent=[edges[0], edges[-1], 0, 1], interpolation="nearest")
        elif GT_edit is None or mask is GT_edit:
            ax.imshow(_row_image(mask, ref_color=REF_C), aspect="auto",
                      extent=[edges[0], edges[-1], 0, 1], interpolation="nearest")
        else:
            ax.imshow(_row_image(mask, GT_edit=GT_edit), aspect="auto",
                      extent=[edges[0], edges[-1], 0, 1], interpolation="nearest")
        crit_mask = (hatch or {}).get(label)
        if crit_mask is None and mask.dtype == bool:
            crit_mask = _CRITICAL_REGISTRY.get(id(mask))
        if crit_mask is not None and crit_mask.any():
            ax.fill_between(t, 0, 1, where=crit_mask, step="post",
                            facecolor="none", edgecolor="black",
                            hatch="///", linewidth=0, antialiased=False)
            has_hatch = True
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(ANON.get(label, label), rotation=0,
                      ha="right", va="center")

    axes[-1].set_xlabel("time (s)")
    if xlim is not None:                  # sharex: one set is enough
        axes[-1].set_xlim(*xlim)

    legend_handles = []
    if GT_edit is not None:
        legend_handles += [Patch(color=TP_C, label="In both edit and GT"),
                          Patch(color=FP_C, label="In edit, not in GT"),
                          Patch(color=FN_C, label="Missed")]
    if has_hatch:
        legend_handles.append(Patch(facecolor="none", edgecolor="black",
                                    hatch="///", label="Marked critical"))
    if legend_handles:
        fig.legend(handles=legend_handles,                  # clear of the x label
                   loc="upper center", bbox_to_anchor=(0.5, -0.06),
                   ncol=3, frameon=False)

    if fname:
        outdir = Path(REPO_ROOT) / "writing" / "Figures" / "sys_eval"
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / fname, dpi=200, bbox_inches="tight")
        print(f"saved {outdir / fname}")
    plt.show()


    





#-----------VOTING ----------DEAD
def labels_dir():
    """100_labels for the active case. Filed at the case root for some cases and
    under 010_source for others, so try both rather than assuming one."""
    for cand in (case_dir() / "100_labels", case_dir() / "010_source" / "100_labels"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"no 100_labels under {case_dir()}")

def load_labels():

    """Every annotator's label file for one case, one dict each.
    GT paper edits share this directory but aren't annotations -- skipped."""
    files = [f for f in sorted(labels_dir().glob("*.json"))
             if not f.name.endswith("_GT_paper_edit.json")]
    if not files:
        raise FileNotFoundError(f"{labels_dir()} is empty")
    return [json.loads(f.read_text()) for f in files]


def importance_votes(labels, bin_s=BIN_S, weights=None):
    """Per-bin count of how many annotators marked that moment as a beat.
    Returns (edges, votes) with len(edges) == len(votes) + 1."""
    #video length check
    durs = {round(l["video_duration_sec"], 2) for l in labels}
    assert len(durs) == 1, f"annotators disagree on duration: {durs}"
    #set number of time bins
    n = int(np.ceil(durs.pop() / bin_s))
    #edges of the time bins (1 extra than n)
    edges = np.arange(n + 1) * bin_s
    #collector for important seconds
    votes = np.zeros(n)
    #do each label file
    for label in labels:
        #scratch voter to prevent 1 person voting twice for a moment
        #as long as the video
        score = np.zeros(n)              
        #analysse eeach beat
        for b in label["beats"]:
            #category weighting can make "cirtical" haeavier if I want 
            w = 1.0 if weights is None else weights.get(b["event_category"], 1.0)
            #rounds down 
            i0 = max(0, int(np.floor(b["start_sec"] / bin_s)))
            #rounds up to make sure that second it counted and makes sure if
            #bins are bigger than 1s to get claimed by a bi
            i1 = min(n, max(i0 + 1, int(np.ceil(b["end_sec"] / bin_s))))  # keep zero-length beats

            score[i0:i1] = np.maximum(score[i0:i1], w)
        votes += score
    n_ann = len(labels)
    votes = votes/ n_ann
    return edges, votes


def consensus_spans(edges, votes, k):
    """Contiguous (start_sec, end_sec) runs where at least k annotators agree."""
    hit = votes >= k
    spans, i = [], 0
    while i < len(hit):
        if not hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(hit) and hit[j + 1]:
            j += 1
        spans.append((float(edges[i]), float(edges[j + 1])))
        i = j + 1
    return spans
    
def score_paper_edit(votes, in_edit, n_ann, tau=None):
    """Grade one paper edit against the human importance curve.
    using w^2 to really emphasise it when the majority think something is important
    ."""
    w = votes
    #multiplier
    g = w**2
    if tau is None:
        tau = (1.0 / n_ann)**2
    bin_s = 1.0 if not hasattr(g, "bin_s") else g.bin_s
    bin_s = BIN_S

    score = float((g[in_edit] - tau).sum())
    best = float(np.maximum(g - tau, 0).sum())
    consensus = w > 0.5                       # strictly more than half, any n_ann
    kept_s = float(in_edit.sum() * bin_s)
    duration = in_edit.sum() * bin_s
    return dict(
        raw_score      = score,
        score_per_s    = score/duration,
        vs_votes       = score / best if best else np.nan,
        recall_w_sq    = float(g[in_edit].sum() / g.sum()) if g.sum() else np.nan,
        junk_frac      = float((g[in_edit] == 0).mean()) if in_edit.any() else np.nan,
        missed_cons_s  = float((consensus & ~in_edit).sum() * bin_s),
        kept_s         = kept_s,
        compression    = kept_s / (len(g) * bin_s),
        tau            = tau,
        n_ann          = n_ann,
    )
def _dilate1(m):
    out = m.copy()
    out[1:] |= m[:-1]
    out[:-1] |= m[1:]
    return out


#--------------------ablations
from collections import Counter


def repeated_frame_counts(paper_edit_json_path):
    """Frame reuse across a paper edit's real beats.

    Returns (unique_frames, repeated_frames) where repeated_frames counts each
    frame once per *extra* appearance: in twice -> 1, in three times -> 2.
    """
    paper_edit = json.loads(Path(paper_edit_json_path).read_text())
    counts = Counter()
    for beat in paper_edit["beats"]:
        s, e = beat.get("start_frame"), beat.get("end_frame")
        if s is None or e is None:
            continue
        counts.update(range(int(s), int(e) + 1))
    total = sum(counts.values())
    unique = len(counts)
    print("Repeated frames" ,total - unique,"\n total frames",unique, "\n %repeated:" , 100*(total - unique) / unique    )
    return total - unique, unique, 100*(total - unique) / unique   


def repeated_frame_pairs(paper_edit_json_path):
    """[(beat_id_a, beat_id_b, n_shared_frames), ...] for overlapping real beats."""
    paper_edit = json.loads(Path(paper_edit_json_path).read_text())
    spans = [(b["beat_id"][0], int(b["start_frame"]), int(b["end_frame"]))
             for b in paper_edit["beats"]
             if b.get("start_frame") is not None and b.get("end_frame") is not None]
    out = []
    for i, (id_a, s_a, e_a) in enumerate(spans):
        for id_b, s_b, e_b in spans[i + 1:]:
            n = min(e_a, e_b) - max(s_a, s_b) + 1
            if n > 0:
                out.append((id_a, id_b, n))
    return out


def GT_paper_edit(case_name, brief=None, n_empty=2, n_map=1):
    """My reference cut, written out in paper-edit shape so it can be scored by
    the same defs as a machine run. Beat bounds come straight from the CSV's own
    src_in_frame/src_out_frame -- no seconds round-trip. Every other beat field
    is null: this carries the cut, not the reasoning. After the cut come
    n_empty fully-null beats, then n_map synthetic MAP beats (archetype and
    requested_flags set, frames null), all numbered in sequence with the rest.

    Writes 100_labels/<csv stem>_GT_paper_edit.json and returns the path."""
    csv_path = GT_cut_path(case_name)
    df = pd.read_csv(csv_path).sort_values("src_in_frame").reset_index(drop=True)

    beats = [{
        "archetype": None,
        "order": i + 1,
        "beat_id": [f"beat-{i + 1:02d}"],
        "segment_type": None,
        "duration_seconds": None,
        "source": None,
        "start_frame": int(row.src_in_frame),
        "end_frame": int(row.src_out_frame),
        "lead_in_seconds": None,
        "search_window_start_seconds": None,
        "search_window_end_seconds": None,
        "requested_flags": None,
        "quote": None,
        "rationale": None,
        "rejected_alternative": None,
        "flag": None,
        "tracked_subject": None,
    } for i, row in df.iterrows()]

    for i in range(len(df), len(df) + n_empty + n_map):
        beat = {k: None for k in beats[0]}
        beat["order"] = i + 1
        beat["beat_id"] = [f"beat-{i + 1:02d}"]
        if i >= len(df) + n_empty:          # the MAP beats come last
            beat["archetype"] = "MAP"
            beat["requested_flags"] = ["BEV_MAP", "RECON_CAM_POSES", "TRACKING"]
        beats.append(beat)

    out_path = csv_path.with_name(f"{csv_path.stem}_GT_paper_edit.json")
    out_path.write_text(json.dumps(
        {"case_name": case_name, "brief": brief, "beats": beats}, indent=2))
    print(f"{case_name}: {len(beats)} beats -> {out_path}")
    return out_path


def _map_beats(paper_edit_json_path):
    """MAP beats of one paper edit, in file order. Looked up by archetype --
    beat_id and order don't survive between two independently written edits."""
    beats = json.loads(Path(paper_edit_json_path).read_text())["beats"]
    return [b for b in beats if b.get("archetype") == "MAP"]


def _map_span(beat):
    """A MAP beat's requested span in seconds. MAP beats are synthetic and carry
    no frames, so the search window is the only span they have."""
    s = beat.get("search_window_start_seconds")
    e = beat.get("search_window_end_seconds")
    return None if s is None or e is None else (float(s), float(e))


def _subject_keys(beat):
    """Tracked subjects of one beat as comparable keys: the Gemini person id
    where there is one, else the descriptor/plain-string text lowercased."""
    keys = set()
    for subj in beat.get("tracked_subject") or []:
        if isinstance(subj, dict):
            pid = subj.get("gem_person_id") or subj.get("person_id")
            keys.add(pid or str(subj.get("descriptor", "")).strip().lower())
        else:
            keys.add(str(subj).strip().lower())
    return keys - {""}


SUBJECT_STOPWORDS = {"the", "a", "an", "his", "her", "their", "with", "in", "on",
                     "of", "and", "wearing", "seen", "person", "man", "woman"}


def _match_subjects(keys_ref, keys_sys):
    """Which subjects the two beats agree on. Gemini person ids must match
    exactly; free text is written independently by each editor ("silver car" vs
    "the silver sedan") so it matches on a shared content word instead.
    Returns (n shared, n unique across both)."""
    def words(k):
        return {w for w in k.replace(",", " ").split() if w not in SUBJECT_STOPWORDS}

    ref_ids = {k for k in keys_ref if k.startswith("person-")}
    sys_ids = {k for k in keys_sys if k.startswith("person-")}
    shared = len(ref_ids & sys_ids)

    free_sys = [k for k in keys_sys if not k.startswith("person-")]
    for key_ref in (k for k in keys_ref if not k.startswith("person-")):
        hit = next((k for k in free_sys if words(key_ref) & words(k)), None)
        if hit is not None:
            free_sys.remove(hit)
            shared += 1
    return shared, len(keys_ref) + len(keys_sys) - shared


def _pair_map_beats(beats_ref, beats_sys):
    """(ref beat, sys beat) pairs, matched on greatest span overlap and falling
    back to file order. Unmatched beats on either side pair with {}."""
    free = list(range(len(beats_sys)))
    pairs = []
    for beat_ref in beats_ref:
        span_ref = _map_span(beat_ref)
        best, best_ov = None, 0.0
        for j in free:
            span_sys = _map_span(beats_sys[j])
            if span_ref and span_sys:
                ov = min(span_ref[1], span_sys[1]) - max(span_ref[0], span_sys[0])
                if ov > best_ov:
                    best, best_ov = j, ov
        if best is None and free:
            best = free[0]
        if best is not None:
            free.remove(best)
        pairs.append((beat_ref, beats_sys[best] if best is not None else {}))
    pairs += [({}, beats_sys[j]) for j in free]      # MAP beats only the system has
    return pairs


def compare_map_beats(paper_edit_ref, paper_edits_sys, label_a="GT",
                      label_b="System", case_name=None, fname=None, edges=None,
                      pad_s=10.0, **plot_kw):
    """MAP beats of one or more system edits, each compared against the same
    reference edit. paper_edits_sys takes a single path, a sequence of paths,
    or {label: path}. label_b names them: one string for a single edit, or one
    string per edit for a sequence -- pass a single string with several edits
    and they fall back to their experiment directory names.

    MAP beats are matched by archetype and greatest span overlap -- never by
    beat_id or order, which are independent between two edits. One row per
    edit: its own beat id, span length and subject count, alongside the
    pair's overlap, span IoU, shared and unique subject counts. Draws each
    pair's spans as bars and returns the DataFrame."""
    if isinstance(paper_edits_sys, dict):
        edits = list(paper_edits_sys.items())
    else:
        paths = ([paper_edits_sys] if isinstance(paper_edits_sys, (str, Path))
                 else list(paper_edits_sys))
        labels = [label_b] if isinstance(label_b, str) else list(label_b)
        assert len(labels) == len(paths), (
            f"label_b has {len(labels)} label(s) for {len(paths)} edits -- "
            f"pass one per edit, e.g. label_b=('System: ablation', 'System: full')")
        edits = list(zip(labels, paths))

    beats_ref = _map_beats(paper_edit_ref)
    scored = {}            # MAP n -> [(label, row) ...], the reference row first
    bars = {}              # MAP n -> [(ylabel, span, span to colour against)]
    for label_sys, path_sys in edits:
        for n, (beat_ref, beat_sys) in enumerate(
                _pair_map_beats(beats_ref, _map_beats(path_sys)), start=1):
            span_ref, span_sys = _map_span(beat_ref), _map_span(beat_sys)
            keys_ref, keys_sys = _subject_keys(beat_ref), _subject_keys(beat_sys)
            n_shared, n_unique = _match_subjects(keys_ref, keys_sys)

            if span_ref and span_sys:
                overlap = max(0.0, min(span_ref[1], span_sys[1]) - max(span_ref[0], span_sys[0]))
                union = max(span_ref[1], span_sys[1]) - min(span_ref[0], span_sys[0])
                iou = overlap / union if union else np.nan
            else:
                overlap, iou = np.nan, np.nan

            # one row per system: the reference is a column, not a row of its own
            scored.setdefault(n, []).append({
                "MAP": n, "editor": label_sys,
                f"{label_a} subjects": len(keys_ref),
                "Subjects": len(keys_sys), "Shared": n_shared,
                "Total": n_unique, "Overlap s": overlap,
                "Span IoU": iou,
            })
            bars.setdefault(n, [(f"{label_a} MAP {n}", span_ref, None)])
            bars[n].append((label_sys, span_sys, span_ref))

    rows = [row for n in sorted(scored) for row in scored[n]]
    df = pd.DataFrame(rows)
    for col in df.columns:                 # whole seconds shouldn't print as 12.0
        if df[col].dtype.kind == "f" and (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype("Int64")
    if len(bars) == 1:
        df = df.drop(columns="MAP")        # only one MAP beat: the column says nothing
    if df.empty:
        print("no MAP beats in either edit")
        return df

    if edges is None and case_name:        # same grid the rest of the notebook uses
        try:
            set_case(case_name)
            edges = time_grid(case_name, labels=load_labels())
        except Exception as e:
            print(f"{case_name}: no time grid ({e}) -- drawing spans on their own axis")

    if edges is not None:      # same figure as every other track in the notebook
        tracks = {label_a: spans_to_mask(
            [s for s in (_map_span(b) for b in beats_ref) if s], edges)}
        for label_sys, path_sys in edits:
            tracks[label_sys] = spans_to_mask(
                [s for s in (_map_span(b) for b in _map_beats(path_sys)) if s], edges)
        # MAP beats are seconds long on a video that runs minutes -- crop to them
        marked = [t for m in tracks.values() for t in edges[:-1][m]]
        xlim = ((max(edges[0], min(marked) - pad_s), min(edges[-1], max(marked) + pad_s))
                if marked else None)
        timeline_agreement(f"{case_name or ''} MAP beats".strip(), edges, tracks,
                           GT_edit=tracks[label_a], fname=fname, xlim=xlim,
                           **plot_kw)      # font sizes etc. live in one place only
        return df

    flat = [bar for n in sorted(bars) for bar in bars[n]]
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.4 * len(flat)))
    for k, (ylabel, span, ref) in enumerate(flat):
        if ref is None:                              # the reference's own row
            if span:
                ax.barh(k, span[1] - span[0], left=span[0], height=0.3, color=REF_C)
            continue
        # every system row is coloured by agreement, same scheme as the timelines
        cuts = sorted({*(span or ()), *ref})
        for lo, hi in zip(cuts, cuts[1:]):
            mid = (lo + hi) / 2
            in_sys = span is not None and span[0] <= mid <= span[1]
            in_ref = ref[0] <= mid <= ref[1]
            colour = TP_C if in_sys and in_ref else FP_C if in_sys else FN_C
            ax.barh(k, hi - lo, left=lo, height=0.6, color=colour)
    ax.set_yticks(range(len(flat)))
    ax.set_yticklabels([b[0] for b in flat])
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.set_title(f"{case_name or ''} MAP beat spans".strip())
    ax.legend(handles=[Patch(color=TP_C, label=f"in both"),
                       Patch(color=FP_C, label=f"in system, not {label_a}"),
                       Patch(color=FN_C, label="missed")],
              frameon=False, ncol=3, loc="upper right")

    if fname:
        outdir = Path(REPO_ROOT) / "writing" / "Figures" / "sys_eval"
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / fname, dpi=200, bbox_inches="tight")
        print(f"saved {outdir / fname}")
    plt.show()
    return df


