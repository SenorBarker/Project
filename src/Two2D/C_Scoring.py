import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt


def sharpness_score(input_frame, ROI):
    '''Gets a focus score for the contents of bounding boxes, which contain the subject being tracked'''
    crop = input_frame[ROI]
    crop_resized = cv2.resize(crop, (64, 64))
    sharpness = cv2.Laplacian(crop_resized, cv2.CV_64F).var()
    return sharpness                                                    


def framing_score(input_frame, ROI):
    h, w, ch = input_frame.shape
    cx_fr = w / 2
    cy_fr = h / 2
    ys_ROI, xs_ROI = np.where(ROI > 0)
    cx_ROI = np.mean(xs_ROI)
    cy_ROI = np.mean(ys_ROI)
    centrality_x = 1 - (np.abs(cx_fr - cx_ROI) / w)
    centrality_y = 1 - (np.abs(cy_fr - cy_ROI) / h)               
    return centrality_x + centrality_y                             


def size_score(frame, ROI):
    h, w, ch = frame.shape                                          
    area_img = h * w
    area_ROI = np.sum(ROI)
    return area_ROI / area_img


def frame_scoring(video_path, frame_range='all', ROIs=None):
    '''This controls all scoring
    INPUTS:
    video_path  : path to the video file to access
    frame_range : in frames (0 being the first frame, to assess); 'all' (default) scores every frame in the video
    ROIs        : the tracking data - bounding boxes or masks - masks preferable
    '''
    print("setting up for frame scoring")
    cap = cv2.VideoCapture(str(video_path))
    ret, test_frame = cap.read()
    h, w, ch = test_frame.shape
    if frame_range == 'all':
        frame_range = range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_size = np.ones((h, w), dtype=np.uint8)

    scores = {}
    for frame_idx in frame_range:
        if frame_idx%100 == 0:
            print(f"scoring frame {frame_idx}")
        ROI = ROIs[frame_idx] if ROIs is not None else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        sharpness_ROI   = sharpness_score(frame, ROI) if ROI is not None else 0
        sharpness_image = sharpness_score(frame, frame_size)
        framing         = framing_score(frame, ROI) if ROI is not None else 0

        if ROI is None:
            visibility   = 0
            subject_size = 0
        else:
            visibility   = 1
            subject_size = size_score(frame, ROI)

        scores[frame_idx] = {
            "sharpness_ROI":   sharpness_ROI,
            "sharpness_image": sharpness_image,
            "framing":         framing,
            "visibility":      visibility,
            "subject_size":    subject_size,
        }
    cap.release()
    return scores

def sequence_scoring(transform_file, frame_scores,time_window):
    '''Takes the FFMPEG analysis of the video and assesses for motion over time
    INPUTS:
    transform_file : path to .trf file
    time_window    : number of frames over which to integrate
    '''
    trf = pd.read_csv(transform_file, sep=' ',
                      names=['frame', 'dx', 'dy', 'rotation', 'zoom', 'extra'],
                      comment='#')

    trf['ang_velocity']   = trf['rotation'].diff()
    trf['ang_accel']      = trf['ang_velocity'].diff()              # fixed: was diff of velocity not rotation
    trf['vel_sign_change'] = (np.sign(trf['ang_velocity']).diff().abs() > 0).astype(int)
    #we need to flip around time - so we look into the future, not the past, thus at frame 1, we know the following window of time will contain these scores
    trf = trf.iloc[::-1].reset_index(drop=True)  # reverse time - cos rolling looks backwards
    
    #calculate each score for all frames at once
    whip        = trf['ang_velocity'].rolling(time_window, min_periods=1).mean().abs().values[::-1]
    double_whip = (trf['ang_accel'].rolling(time_window, min_periods=1).mean().abs() *
                   trf['vel_sign_change'].rolling(time_window, min_periods=1).max()).values[::-1]
    wobble      = trf['ang_accel'].abs().rolling(time_window, min_periods=1).mean().values[::-1]
    steps       = trf['dy'].diff().abs().rolling(time_window, min_periods=1).mean().values[::-1]
    sway        = trf['dx'].diff().abs().rolling(time_window, min_periods=1).mean().values[::-1]
    frame_indices = trf['frame'].values[::-1]  # original frame numbers, flipped back
    
    #take the FRAME SCORES and make a pandas seriesy, then reverse, then convonvolve with rolling - it handles padding better then np's convolve
    # integrate each frame score component forward
    integrated_frames = {}
    for key in ['visibility', 'sharpness_ROI', 'sharpness_image', 'framing', 'subject_size']:
        vals = pd.Series([frame_scores[i][key] for i in frame_indices])
        integrated_frames[key] = vals.iloc[::-1].rolling(time_window, min_periods=1).mean().iloc[::-1].values
       
    
    #then write by frame for comparison with the frame scoring
    seq_scores = {
        frame_idx: {
        'whip':             whip[i],
        'double_whip':      double_whip[i],
        'wobble':           wobble[i],
        'steps':            steps[i],
        'sway':             sway[i],
        'visibility':       integrated_frames['visibility'][i],
        'sharpness_ROI':    integrated_frames['sharpness_ROI'][i],
        'sharpness_image':  integrated_frames['sharpness_image'][i],
        'framing':          integrated_frames['framing'][i],
        'subject_size':     integrated_frames['subject_size'][i],
    
            }
        for i, frame_idx in enumerate(frame_indices)
    }
    return seq_scores

def score_formula(seq_scores, params):
    '''the decision on what to weight each individual score is done in here'''
    w = params["weights"]

    frame_final_scores = []
    for frame_idx in seq_scores:
        visibility      = seq_scores[frame_idx]["visibility"]
        sharpness_image = seq_scores[frame_idx]["sharpness_image"]
        sharpness_ROI   = seq_scores[frame_idx]["sharpness_ROI"]
        framing         = seq_scores[frame_idx]["framing"]
        subject_size    = seq_scores[frame_idx]["subject_size"]
        whip            = seq_scores[frame_idx]["whip"]
        double_whip     = seq_scores[frame_idx]["double_whip"]
        wobble          = seq_scores[frame_idx]["wobble"]
        steps           = seq_scores[frame_idx]["steps"]
        sway            = seq_scores[frame_idx]["sway"]

        frame_total_score = visibility * (
            w["sharpness_ROI"]   * sharpness_ROI +
            w["sharpness_image"] * sharpness_image +
            w["framing"]         * framing +
            w["subject_size"]    * subject_size
        )
        sequence_total_score = (
            w["whip"]        * whip +
            w["double_whip"] * double_whip +
            w["wobble"]      * wobble +
            w["steps"]       * steps +
            w["sway"]        * sway
        )
        final_score = frame_total_score / (1 + sequence_total_score)
        frame_final_scores.append(final_score)

    return frame_final_scores                                        


def frame_prob_dist(frame_scores: dict, chart=False) -> dict:
    """
    Takes the total score per frame and returns a PDF for each camera
    frame_scores: {'cam_A': 0.82, 'cam_B': 0.45, 'cam_C': 0.61}
    returns:      {'cam_A': 0.54, 'cam_B': 0.12, 'cam_C': 0.34}
    """
    cameras = list(frame_scores.keys())
    scores  = np.array(list(frame_scores.values()))
    exp_s   = np.exp(scores - scores.max()) # soft max
    probs   = exp_s / exp_s.sum()

    if chart:
        fig, ax = plt.subplots()
        ax.bar(cameras, probs)
        ax.set_ylabel('Probability')
        plt.show()

    return dict(zip(cameras, probs))

#I need a stitcher to put all cams together and align them here

def editor(cam_scores: dict, time_window: int, cutaway_gap: int):
    # best overall = default: is subject in view (score is above 0), then what's the average?
    #all cams have the same duration. missing frames will have "nones"
    mean_quality = {cam: np.mean(np.array(scores) > 0) for cam, scores in cam_scores.items()}
    default_cam = max(mean_quality, key=mean_quality.get)
    
    # per frame pdf
    n_frames = len(next(iter(cam_scores.values())))
    per_frame_pdf = []
    for i in range(n_frames):
        frame_scores = {cam: cam_scores[cam][i] for cam in cam_scores}
        per_frame_pdf.append(frame_prob_dist(frame_scores))
    
    # state machine
    edit = [] # will be a per-frame camera list. 
    state = 'DEFAULT'
    current_cam = default_cam
    hold_until = 0

    for i, pdf in enumerate(per_frame_pdf):
        best_this_frame = max(pdf, key=pdf.get)
        
        if i < hold_until:
            edit.append(current_cam)
            continue
        
        if state == 'DEFAULT':
            if best_this_frame != default_cam:
                current_cam = best_this_frame
                state = 'CUTAWAY'
                hold_until = i + time_window
            edit.append(current_cam)
        
        elif state == 'CUTAWAY':
            if best_this_frame == default_cam or pdf[default_cam] > pdf[current_cam]:
                current_cam = default_cam
                state = 'DEFAULT'
                hold_until = i + time_window
            edit.append(current_cam)
    
    return default_cam, per_frame_pdf, edit




