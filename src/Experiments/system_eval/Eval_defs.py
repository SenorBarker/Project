
from A_Config import set_case, case_dir
import json
from pathlib import Path
from B_video_processing import frame_times_at_indices
from A_Config import set_case, source_video_path, source_dir
import numpy as np
import pandas as pd
BIN_S = 1.0 

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



def annotator_beats_to_grid(label, edges):
    """One annotator's own selection, as a kept/not mask."""
    spans = [(b["start_sec"], b["end_sec"]) for b in label["beats"]]
    return spans_to_mask(spans, edges)



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
    Returns the mean baseline row plus, for each reported measure, the 5th/95th
    percentile of the null and where the real edit sits inside it."""
    rng = np.random.default_rng(seed)
    n = len(paper_edit)
    d = np.diff(np.concatenate(([0], paper_edit.view(np.int8), [0])))
    lens = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)

    draws = []
    for _ in range(n_draws):
        m = np.zeros(n, dtype=bool)
        for L in sorted(rng.permutation(lens), reverse=True):   # longest first
            for _ in range(50):
                i = rng.integers(0, max(1, n - L))
                if not m[i:i + L].any():
                    m[i:i + L] = True
                    break
        draws.append(score_paper_vs_GT_edits(GT_edit, m))

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
    return row, pd.DataFrame(stats).T.round(3)




#-----------VOTING ----------DEAD
def labels_dir():
    """100_labels for the active case. Filed at the case root for some cases and
    under 010_source for others, so try both rather than assuming one."""
    for cand in (case_dir() / "100_labels", case_dir() / "010_source" / "100_labels"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"no 100_labels under {case_dir()}")

def load_labels():

    """Every annotator's label file for one case, one dict each."""
    files = sorted(labels_dir().glob("*.json"))
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