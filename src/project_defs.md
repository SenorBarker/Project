# All defs in `src/`

326 defs across 42 modules. Labels are full signatures.
Rectangles are functions, `[[double]]` are classes, `([rounded])` are methods of the class above them.
Excludes `Models/` (vendored), `Dead_Code/`, and the Experiments/Exploration notebooks.

Call relationships live in [`project_callgraph.mmd`](project_callgraph.mmd).

```mermaid
%% All defs in /workspace/project/src — grouped by role, then by module.
%% Labels are full signatures. No call edges: see project_callgraph.mmd for those.
flowchart LR
  subgraph G_Config___report_plumbing["Config & report plumbing"]
    direction LR
    subgraph M_A_Config_py["A_Config.py"]
      direction TB
      M_A_Config_py_0["set_case(case_name, experiment)"]
      M_A_Config_py_1["case_name()"]
      M_A_Config_py_2["case_dir()"]
      M_A_Config_py_3["assets_dir()"]
      M_A_Config_py_4["asset_name()"]
      M_A_Config_py_5["report_path()"]
      M_A_Config_py_6["asset_list_path()"]
      M_A_Config_py_7["_with_experiment(base)"]
      M_A_Config_py_8["source_dir()"]
      M_A_Config_py_9["source_video_path()"]
      M_A_Config_py_10["query_dir()"]
      M_A_Config_py_11["agent_p_output_dir()"]
      M_A_Config_py_12["yolo_masks_dir()"]
      M_A_Config_py_13["sam3_masks_dir()"]
      M_A_Config_py_14["frames_for_cam_poses_dir()"]
      M_A_Config_py_15["frames_for_recon_dir()"]
      M_A_Config_py_16["frames_for_recon_dir()"]
      M_A_Config_py_17["recon_for_MAP_dir()"]
      M_A_Config_py_18["recon_for_EVENT_dir()"]
      M_A_Config_py_19["mega_SAM_output_dir()"]
      M_A_Config_py_20["vggt_o_output_dir()"]
      M_A_Config_py_21["predictions_path()"]
      M_A_Config_py_22["cut3r_output_dir()"]
      M_A_Config_py_23["lingbot_map_dir()"]
      M_A_Config_py_24["ply_dir()"]
      M_A_Config_py_25["reg_dir()"]
      M_A_Config_py_26["gps_data_dir()"]
      M_A_Config_py_27["csv_path_gps()"]
      M_A_Config_py_28["csv_path_pose()"]
      M_A_Config_py_29["has_gps_data()"]
      M_A_Config_py_30["length_units()"]
      M_A_Config_py_31["to_report_path(path)"]
      M_A_Config_py_32["to_repo_path(path)"]
    end
    subgraph M_C_CSV_report_py["C_CSV_report.py"]
      direction TB
      M_C_CSV_report_py_33["_add_rows(path, data)"]
      M_C_CSV_report_py_34["add_to_report(data)"]
      M_C_CSV_report_py_35["add_to_asset_list(data)"]
      M_C_CSV_report_py_36["frames_to_time(frames, fps, timecode)"]
      M_C_CSV_report_py_37["attach_times_of_day(video_path, fps)"]
    end
    subgraph M_A_Interactions_py["A_Interactions.py"]
      direction TB
      M_A_Interactions_py_38["subject_selector(video_path, detections, model_path, conf)"]
    end
  end
  subgraph G_2D_analysis__Gemini__video__scoring["2D analysis: Gemini, video, scoring"]
    direction LR
    subgraph M_Two2D_A_gemini_py["Two2D/A_gemini.py"]
      direction TB
      M_Two2D_A_gemini_py_39["gemini_vid_to_text(video_path, case_name, query_dir)"]
      M_Two2D_A_gemini_py_40["query_from_description(operator_prompt, description_path, objects_path, case_dir)"]
      M_Two2D_A_gemini_py_41["gemini_cache_video(video_path, query_dir, ttl)"]
      M_Two2D_A_gemini_py_42["gemini_query_CSV_cached(prompt, asset_name, CSV_path, query_dir, extend_ttl)"]
    end
    subgraph M_Two2D_A_gemini_v02_py["Two2D/A_gemini_v02.py"]
      direction TB
      M_Two2D_A_gemini_v02_py_43["get_box(det)"]
      M_Two2D_A_gemini_v02_py_44["draw_point_overlay(dets, native_frame, out_path)"]
      M_Two2D_A_gemini_v02_py_45["_deterministic_config(cache_name)"]
      M_Two2D_A_gemini_v02_py_46["_log_response_meta(response, label)"]
      M_Two2D_A_gemini_v02_py_47["gemini_vid_to_text(video_path, case_name, query_dir)"]
      M_Two2D_A_gemini_v02_py_48["query_from_description(operator_prompt, description_path, objects_path, case_dir)"]
      M_Two2D_A_gemini_v02_py_49["gemini_cache_video(video_path, query_dir, ttl)"]
      M_Two2D_A_gemini_v02_py_50["gemini_query_CSV_cached(prompt, asset_name, CSV_path, query_dir, extend_ttl)"]
    end
    subgraph M_Two2D_A_gemini_v03_py["Two2D/A_gemini_v03.py"]
      direction TB
      M_Two2D_A_gemini_v03_py_51["get_box(det)"]
      M_Two2D_A_gemini_v03_py_52["draw_point_overlay(dets, native_frame, out_path)"]
      M_Two2D_A_gemini_v03_py_53["_deterministic_config(cache_name)"]
      M_Two2D_A_gemini_v03_py_54["_strip_code_fence(text)"]
      M_Two2D_A_gemini_v03_py_55["_log_response_meta(response, label)"]
      M_Two2D_A_gemini_v03_py_56["gemini_vid_to_text(video_path, case_name, query_dir)"]
      M_Two2D_A_gemini_v03_py_57["query_from_description(operator_prompt, description_path, objects_path, case_dir)"]
      M_Two2D_A_gemini_v03_py_58["gemini_cache_video(video_path, query_dir, ttl)"]
      M_Two2D_A_gemini_v03_py_59["gemini_query_CSV_cached(prompt, asset_name, CSV_path, query_dir, extend_ttl)"]
    end
    subgraph M_Two2D_B_video_processing_py["Two2D/B_video_processing.py"]
      direction TB
      M_Two2D_B_video_processing_py_60["_blur_score(gray)"]
      M_Two2D_B_video_processing_py_61["video_dims(video_path)"]
      M_Two2D_B_video_processing_py_62["image_sequencer(video_path, output_dir, res, interval_sec, start_frame, end_frame, search_window, min_sharpness, outrank_margin, score_scale, output_ext, jpeg_quality)"]
      M_Two2D_B_video_processing_py_63["main()"]
      M_Two2D_B_video_processing_py_64["_probe_format(video_path)"]
      M_Two2D_B_video_processing_py_65["video_creation_time(video_path)"]
      M_Two2D_B_video_processing_py_66["video_start_time(video_path)"]
      M_Two2D_B_video_processing_py_67["video_fps(video_path)"]
      M_Two2D_B_video_processing_py_68["frames_at_indices(video_path, frame_indices)"]
      M_Two2D_B_video_processing_py_69["seek_exact(cap, frame_idx)"]
      M_Two2D_B_video_processing_py_70["frames_at_times(video_path, target_times_s)"]
      M_Two2D_B_video_processing_py_71["available_frames(frames_dir, name_fmt)"]
    end
    subgraph M_Two2D_C_Scoring_py["Two2D/C_Scoring.py"]
      direction TB
      M_Two2D_C_Scoring_py_72["sharpness_score(input_frame, ROI)"]
      M_Two2D_C_Scoring_py_73["framing_score(input_frame, ROI)"]
      M_Two2D_C_Scoring_py_74["size_score(frame, ROI)"]
      M_Two2D_C_Scoring_py_75["frame_scoring(video_path, frame_range, ROIs)"]
      M_Two2D_C_Scoring_py_76["sequence_scoring(transform_file, frame_scores, time_window)"]
      M_Two2D_C_Scoring_py_77["score_formula(seq_scores, params)"]
      M_Two2D_C_Scoring_py_78["frame_prob_dist(frame_scores, chart)"]
      M_Two2D_C_Scoring_py_79["editor(cam_scores, time_window, cutaway_gap)"]
    end
    subgraph M_Two2D_AX_gem_SAM_matcher_py["Two2D/AX_gem_SAM_matcher.py"]
      direction TB
      M_Two2D_AX_gem_SAM_matcher_py_80["gem_box_to_xyxy(gem_box, width, height)"]
      M_Two2D_AX_gem_SAM_matcher_py_81["sam_det_to_xyxy(sam_det, vid_w, vid_h)"]
      M_Two2D_AX_gem_SAM_matcher_py_82["iou_xyxy(box_a, box_b)"]
      M_Two2D_AX_gem_SAM_matcher_py_83["load_SAM_dets(raw_detection_rows, SAM_dets_path)"]
      M_Two2D_AX_gem_SAM_matcher_py_84["gem_person_targets_lookup(fps, paper_edit_json_path, gem_people_json_path)"]
      M_Two2D_AX_gem_SAM_matcher_py_85["match_gem_people_to_sam(vid_w, vid_h, fps, gem_person_targets, SAM_dets, iou_threshold)"]
    end
    subgraph M_Two2D_AY_claude_crop_matcher_py["Two2D/AY_claude_crop_matcher.py"]
      direction TB
      M_Two2D_AY_claude_crop_matcher_py_86["descriptor_targets(paper_edit_json_path)"]
      M_Two2D_AY_claude_crop_matcher_py_87["_mask_frames(track_dir)"]
      M_Two2D_AY_claude_crop_matcher_py_88["_isolate(frame, mask, pad, max_side)"]
      M_Two2D_AY_claude_crop_matcher_py_89["_save_8bit(bgr, out_path, colors)"]
      M_Two2D_AY_claude_crop_matcher_py_90["write_masked_crops(masks_root, video_path, out_dir, per_track, min_mask_px, subject_slug)"]
      M_Two2D_AY_claude_crop_matcher_py_91["_match_prompt(track_key, crop_paths, candidates, targets)"]
      M_Two2D_AY_claude_crop_matcher_py_92["_ask_sync(prompt, cwd, model)"]
      M_Two2D_AY_claude_crop_matcher_py_93["_ask(prompt, cwd, model)"]
      M_Two2D_AY_claude_crop_matcher_py_94["match_tracks_to_descriptors(crops_by_track, targets, masks_root, fps, model, delete_crops)"]
    end
    subgraph M_Two2D_W_video_editor_py["Two2D/W_video_editor.py"]
      direction TB
      M_Two2D_W_video_editor_py_95["video_edit(video_path, start, end, out_path, handles)"]
      M_Two2D_W_video_editor_py_96["mask_frame_range(masks_dir)"]
      M_Two2D_W_video_editor_py_97["_load_mask_bool(mask_dir, frame_idx)"]
      M_Two2D_W_video_editor_py_98["apply_mask_overlay(frame_bgr, mask_bool, color, alpha)"]
      M_Two2D_W_video_editor_py_99["video_overlay_edit(video_path, masks_dir, start, end, mask_start, mask_end, handles, color, alpha, name)"]
      M_Two2D_W_video_editor_py_100["mask_compositor(video_path, start, end, masks_dir, mask_start, mask_end, handles, color, alpha)"]
      M_Two2D_W_video_editor_py_101["_mask_overlay_mezzanine_clip(video_path, fps, start_frame, end_frame, masks_dirs, out_dir, name)"]
      M_Two2D_W_video_editor_py_102["_real_mezzanine_clip(beat, video_path, fps, out_dir)"]
      M_Two2D_W_video_editor_py_103["_map_mezzanine_clip(beat, type, target_w, target_h, target_fps, out_dir)"]
      M_Two2D_W_video_editor_py_104["_audio_duration(clip_path)"]
      M_Two2D_W_video_editor_py_105["_concat_mezzanine_clips(clip_paths, out_path, fade_s)"]
      M_Two2D_W_video_editor_py_106["assemble_paper_edit(video_path, analysis_2d_for_decisions)"]
      M_Two2D_W_video_editor_py_107["find_true_audio_span(video_path, window_start_s, window_end_s, analysis_margin_s, noise_db, min_silence_s, handle_s)"]
      M_Two2D_W_video_editor_py_108["find_cut_point_in_zone(zone_lo, zone_hi, boundary_frames, prefer_near)"]
      M_Two2D_W_video_editor_py_109["find_cut_points(beat, video_path, fps, analysis_2d_for_decisions, handle_s, analysis_margin_s, noise_db, min_silence_s)"]
      M_Two2D_W_video_editor_py_110["find_all_cut_points(video_path, analysis_2d_for_decisions, **kwargs)"]
    end
    subgraph M_D_2d_analysis_py["D_2d_analysis.py"]
      direction TB
      M_D_2d_analysis_py_111["frames_present(masks_path)"]
      M_D_2d_analysis_py_112["list_track_id_dirs(masks_root, subject_slug)"]
      M_D_2d_analysis_py_113["is_3D_possible(frames_list, recon_tool)"]
      M_D_2d_analysis_py_114["contiguous_durations(frames, tolerance)"]
      M_D_2d_analysis_py_115["GPS_test(video_path, api_key)"]
      M_D_2d_analysis_py_116["analysis_2D(video_path, api_key)"]
      M_D_2d_analysis_py_117["mask_analysis(masks_dir, report_prefix)"]
      M_D_2d_analysis_py_118["analysis_2d_from_masks(paper_edit_json_path)"]
      M_D_2d_analysis_py_119["analysis_2d_gemvsSAM(gem_person_targets, SAM_dets, match_results, tolerance)"]
    end
  end
  subgraph G_Model_runners["Model runners"]
    direction LR
    subgraph M_run_models_A_LOCAL_SAM3_py["run_models/A_LOCAL_SAM3.py"]
      direction TB
      M_run_models_A_LOCAL_SAM3_py_120[["class ObjectTooDense"]]
      M_run_models_A_LOCAL_SAM3_py_121["get_predictor()"]
      M_run_models_A_LOCAL_SAM3_py_122["release_predictor()"]
      M_run_models_A_LOCAL_SAM3_py_123["_track_span(predictor, clip_path, subject, threshold, expected_count)"]
      M_run_models_A_LOCAL_SAM3_py_124["_run_sam3_tracking_local(video_path, tracking_requests_by_subject, threshold, *_roboflow_args, **_roboflow_kwargs)"]
    end
    subgraph M_run_models_A_ROBOFLOW_SAM3_py["run_models/A_ROBOFLOW_SAM3.py"]
      direction TB
      M_run_models_A_ROBOFLOW_SAM3_py_125["find_preceding_keyframe_s(video_path, target_s, initial_window_s, max_window_s)"]
      M_run_models_A_ROBOFLOW_SAM3_py_126["frame_pts_times(video_path)"]
      M_run_models_A_ROBOFLOW_SAM3_py_127["frame_index_at_time(video_path, t_s)"]
      M_run_models_A_ROBOFLOW_SAM3_py_128["make_span_clip(video_path, out_path, start_s, end_s)"]
      M_run_models_A_ROBOFLOW_SAM3_py_129["unwrap_predictions(raw)"]
      M_run_models_A_ROBOFLOW_SAM3_py_130["decode_mask_from_detection(det, image_shape)"]
      M_run_models_A_ROBOFLOW_SAM3_py_131["extract_track_id(det)"]
      M_run_models_A_ROBOFLOW_SAM3_py_132["_slugify_subject(subject)"]
      M_run_models_A_ROBOFLOW_SAM3_py_133["_dispatch_sam3_tracking(*args, **kwargs)"]
      M_run_models_A_ROBOFLOW_SAM3_py_134["track_subject_sam3(video_path, threshold, requested_region, requested_plan)"]
      M_run_models_A_ROBOFLOW_SAM3_py_135["run_sam3_manual(video_path, subject, start_frame, end_frame, threshold, requested_region, requested_plan)"]
      M_run_models_A_ROBOFLOW_SAM3_py_136["_run_sam3_tracking(video_path, tracking_requests_by_subject, threshold, requested_region, requested_plan)"]
    end
    subgraph M_run_models_A_YOLO_seg_py["run_models/A_YOLO_seg.py"]
      direction TB
      M_run_models_A_YOLO_seg_py_137["_get_subject_ids(model, subjects)"]
      M_run_models_A_YOLO_seg_py_138["_detect_in_frame(model, cap, frame_idx, subject_ids, h, w, conf)"]
      M_run_models_A_YOLO_seg_py_139["_expand_bounds(model, cap, total_frames, subject_ids, h, w, hint_start, hint_end, miss_threshold, conf)"]
      M_run_models_A_YOLO_seg_py_140["_run_tracking(model, cap, out_dir, subject_ids, h, w, start_frame, end_frame, tracker, conf)"]
      M_run_models_A_YOLO_seg_py_141["track_subject_masks(video_path, output_dir, subjects, start_frame, end_frame, model_path, tracker, conf)"]
      M_run_models_A_YOLO_seg_py_142["track_subject_masks_from_hints(video_path, output_dir, detections, miss_threshold, model_path, tracker, conf)"]
    end
    subgraph M_run_models_B_lingbot_map_py["run_models/B_lingbot_map.py"]
      direction TB
      M_run_models_B_lingbot_map_py_143["_import_lingbot_demo()"]
      M_run_models_B_lingbot_map_py_144["main()"]
      M_run_models_B_lingbot_map_py_145["_frame_npz_path(out_dir, frame_num)"]
      M_run_models_B_lingbot_map_py_146["save_lingbot_frame(out_dir, frame_num, **arrays)"]
      M_run_models_B_lingbot_map_py_147["load_lingbot_frame(out_dir, frame_num)"]
      M_run_models_B_lingbot_map_py_148["run_lingbot_map(mode, first_k, stride, launch_viewer, port, **demo_kwargs)"]
    end
    subgraph M_run_models_E_VGGT_O_shards_py["run_models/E_VGGT_O_shards.py"]
      direction TB
      M_run_models_E_VGGT_O_shards_py_149["discover_vggt_omega_shards(output_dir)"]
      M_run_models_E_VGGT_O_shards_py_150["run_vggt_omega_shards_batched(image_dir, output_dir, checkpoint_path, vggt_omega_shards_dir, devices, image_resolution, conf_thres, max_points, show_cam, batch_size)"]
      M_run_models_E_VGGT_O_shards_py_151["run_vggt_omega_shards(image_dir, output_dir, checkpoint_path, vggt_omega_shards_dir, devices, image_resolution, conf_thres, max_points, show_cam, frame_range, output_name)"]
    end
    subgraph M_run_models_E_VGGT_omega_py["run_models/E_VGGT_omega.py"]
      direction TB
      M_run_models_E_VGGT_omega_py_152["_unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)"]
      M_run_models_E_VGGT_omega_py_153["run_vggt_omega(image_dir, output_dir, checkpoint_path, vggt_omega_dir, image_resolution, conf_thres, max_points, show_cam)"]
    end
    subgraph M_run_models_E_cut3r_recon_py["run_models/E_cut3r_recon.py"]
      direction TB
      M_run_models_E_cut3r_recon_py_154["_make_raymap(c2w, h, w, intrinsics)"]
      M_run_models_E_cut3r_recon_py_155["_save_ply(path, pts3d_world, colors, confidence)"]
      M_run_models_E_cut3r_recon_py_156["_export_outputs(output_dir, f_id, depth, conf, color, c2w, intrin, pts3d_world, R_norm, t_norm, nvs_rgb)"]
      M_run_models_E_cut3r_recon_py_157["_third_person_c2w(c2w, back_dist, up_dist)"]
      M_run_models_E_cut3r_recon_py_158["run_cut3r(frames_dir, output_dir, ckpt_path, size, device, render_3rdperson, back_dist, up_dist, revisit)"]
    end
    subgraph M_run_models_E_megasam_recon_py["run_models/E_megasam_recon.py"]
      direction TB
      M_run_models_E_megasam_recon_py_159["_clean_env(extra)"]
      M_run_models_E_megasam_recon_py_160["_run(cmd, cwd, env, desc)"]
      M_run_models_E_megasam_recon_py_161["run_megasam(mega_sam_dir, scene_name, conda_env)"]
    end
    subgraph M_run_models_E_megasam_vis_py["run_models/E_megasam_vis.py"]
      direction TB
      M_run_models_E_megasam_vis_py_162["visualise_megasam(npz_path, n_samples)"]
    end
    subgraph M_run_models_E_realityscan_recon_py["run_models/E_realityscan_recon.py"]
      direction TB
      M_run_models_E_realityscan_recon_py_163["_run(cmd, cwd, desc)"]
      M_run_models_E_realityscan_recon_py_164["run_realityscan(frames_dir, output_dir, rs_exe, project_name)"]
    end
    subgraph M_run_models_spike_sam3_box_seed_py["run_models/spike_sam3_box_seed.py"]
      direction TB
      M_run_models_spike_sam3_box_seed_py_165["box_2d_to_corners(box_2d, width, height)"]
      M_run_models_spike_sam3_box_seed_py_166["main()"]
    end
    subgraph M_run_models_spike_sam3_local_py["run_models/spike_sam3_local.py"]
      direction TB
      M_run_models_spike_sam3_local_py_167["main()"]
    end
  end
  subgraph G_3D__recon_objects___transforms["3D: recon objects & transforms"]
    direction LR
    subgraph M_Thr3D_F_cut3r_vis_py["Thr3D/F_cut3r_vis.py"]
      direction TB
      M_Thr3D_F_cut3r_vis_py_168["load_ply(path)"]
      M_Thr3D_F_cut3r_vis_py_169["visualise_cut3r(output_dir)"]
    end
    subgraph M_Thr3D_F_post_recon_processing_py["Thr3D/F_post_recon_processing.py"]
      direction TB
      M_Thr3D_F_post_recon_processing_py_170["load_reality_scan_trace(csv_path)"]
      M_Thr3D_F_post_recon_processing_py_171["load_cut3r_trace(camera_dir)"]
      M_Thr3D_F_post_recon_processing_py_172["load_cut3r_trace_v2(camera_dir, full_pose)"]
      M_Thr3D_F_post_recon_processing_py_173["load_megasam_trace(npz_path, frames_dir, full_pose)"]
      M_Thr3D_F_post_recon_processing_py_174["load_VGGT_O_trace(npz_path, frames_dir)"]
      M_Thr3D_F_post_recon_processing_py_175["load_VGGT_trace(sparse_reconstruction_dir, frames_dir, full_pose)"]
      M_Thr3D_F_post_recon_processing_py_176["load_lingbot_map_trace(output_dir, full_pose)"]
      M_Thr3D_F_post_recon_processing_py_177["load_VGGT_O_predictions(npz_path)"]
      M_Thr3D_F_post_recon_processing_py_178[["class Reconstruction"]]
      M_Thr3D_F_post_recon_processing_py_179(["load(cls, frames_dir, preds_path, frame_name_fmt, frame_range)"])
      M_Thr3D_F_post_recon_processing_py_180(["_cam0_anchor(self)"])
      M_Thr3D_F_post_recon_processing_py_181(["cam_pos_dict(self, full_pose)"])
      M_Thr3D_F_post_recon_processing_py_182(["VGGT_O_preds_to_ply_export(self, out_dir, R, t, s, multi, centre_to_cam00)"])
      M_Thr3D_F_post_recon_processing_py_183["_write_ply(path, points, colors, confidence)"]
      M_Thr3D_F_post_recon_processing_py_184["build_frame_index(frames_dir, name_fmt, frame_range)"]
      M_Thr3D_F_post_recon_processing_py_185["positions_to_frame_dict(positions, frames_dir, frame_name_fmt)"]
    end
    subgraph M_Thr3D_F_transpose_to_recon_objects_py["Thr3D/F_transpose_to_recon_objects.py"]
      direction TB
      M_Thr3D_F_transpose_to_recon_objects_py_186["_as_hw(arr)"]
      M_Thr3D_F_transpose_to_recon_objects_py_187["_as_nhw(arr)"]
      M_Thr3D_F_transpose_to_recon_objects_py_188["_load_lowres_rgb_from_full(frames_dir, frame_num, low_h, low_w)"]
      M_Thr3D_F_transpose_to_recon_objects_py_189["cut3r_to_reconstruction(frames_dir, cut3r_output_dir)"]
      M_Thr3D_F_transpose_to_recon_objects_py_190["megasam_to_reconstruction(frames_dir, npz_path)"]
      M_Thr3D_F_transpose_to_recon_objects_py_191["lingbot_map_to_reconstruction(frames_dir, lingbot_output_dir)"]
      M_Thr3D_F_transpose_to_recon_objects_py_192["vggt_plain_to_reconstruction(frames_dir, vggt_output_dir)"]
    end
    subgraph M_Thr3D_G_transforms_alignments_py["Thr3D/G_transforms_alignments.py"]
      direction TB
      M_Thr3D_G_transforms_alignments_py_193["transform_RST(positions, R, s, t)"]
      M_Thr3D_G_transforms_alignments_py_194["combine_transforms(R_cb, s_cb, t_cb, R_ba, s_ba, t_ba)"]
      M_Thr3D_G_transforms_alignments_py_195["unproject(depth_full, intrinsic, extrinsic)"]
      M_Thr3D_G_transforms_alignments_py_196["apply_cam0_frame(positions, R0, origin)"]
      M_Thr3D_G_transforms_alignments_py_197["_rot_x(deg)"]
      M_Thr3D_G_transforms_alignments_py_198["_rot_y(deg)"]
      M_Thr3D_G_transforms_alignments_py_199["_rot_z(deg)"]
      M_Thr3D_G_transforms_alignments_py_200["_rc_rotation_matrix(yaw, pitch, roll)"]
      M_Thr3D_G_transforms_alignments_py_201["ortho_charts(dataA, label_A, dataB, label_B, title, out_path, rmse, mean_dist)"]
      M_Thr3D_G_transforms_alignments_py_202["recon_to_recon_matcher(source_dict, target_dict, max_frame_gap)"]
      M_Thr3D_G_transforms_alignments_py_203["recon_to_recon_transform(recon_from, cam_poses_to_dict)"]
      M_Thr3D_G_transforms_alignments_py_204["merge_pos(*sub_pos_real_dicts)"]
      M_Thr3D_G_transforms_alignments_py_205["_umeyama_solve(A, B, ref_A, ref_B, with_scale, up_A, up_B, up_weight)"]
      M_Thr3D_G_transforms_alignments_py_206["_umeyama_finish(A, B, R, s, t, label_A, label_B, out_path)"]
      M_Thr3D_G_transforms_alignments_py_207["umeyama_align(A, B, label_A, label_B, out_path, with_scale, up_A, up_B, up_weight, skip_indices)"]
      M_Thr3D_G_transforms_alignments_py_208["umeyama_align_anchor(A, B, label_A, label_B, out_path, with_scale, up_A, up_B, up_weight, anchor_index, skip_indices)"]
    end
    subgraph M_Thr3D_U_scenepic_o3d_worker_py["Thr3D/U_scenepic_o3d_worker.py"]
      direction TB
      M_Thr3D_U_scenepic_o3d_worker_py_209["view_ply_sequence_with_scenepic(ply_dir, output_html, point_size, pattern)"]
    end
  end
  subgraph G_Geo__metric___subject_space["Geo, metric & subject space"]
    direction LR
    subgraph M_H_Geolocation_GMAPS_py["H_Geolocation_GMAPS.py"]
      direction TB
      M_H_Geolocation_GMAPS_py_210["fetch_metadata(lat, lon, api_key)"]
      M_H_Geolocation_GMAPS_py_211["fetch_image(lat, lon, heading, api_key, size, fov, pitch)"]
      M_H_Geolocation_GMAPS_py_212["fetch_pano(lat, lon, api_key, headings, case_name)"]
      M_H_Geolocation_GMAPS_py_213["show_pano(candidates, title)"]
      M_H_Geolocation_GMAPS_py_214["compute_sift(image, mask_watermark, use_colour)"]
      M_H_Geolocation_GMAPS_py_215["add_sift_to_candidates(candidates, use_colour)"]
      M_H_Geolocation_GMAPS_py_216["match_sift(des_a, des_b, ratio)"]
      M_H_Geolocation_GMAPS_py_217["fetch_panos(lat, lon, api_key, step_m, headings, case_name)"]
      M_H_Geolocation_GMAPS_py_218["match_pano_sets(*panos, use_colour)"]
      M_H_Geolocation_GMAPS_py_219["cross_match_table(match_results)"]
      M_H_Geolocation_GMAPS_py_220["cross_match_table2(match_results)"]
      M_H_Geolocation_GMAPS_py_221["show_matches(results, direction, heading_c, heading_n)"]
      M_H_Geolocation_GMAPS_py_222["show_matches_tb(results, direction, heading_c, heading_n)"]
      M_H_Geolocation_GMAPS_py_223["show_frame_matches(img1, img2, pts1, pts2, title, max_lines)"]
      M_H_Geolocation_GMAPS_py_224["rotation_matrix(h_deg)"]
      M_H_Geolocation_GMAPS_py_225["filter_epipolar(match_results, threshold)"]
      M_H_Geolocation_GMAPS_py_226["triangulate_matches(filtered_results)"]
      M_H_Geolocation_GMAPS_py_227["localise_frame(frame_path, points_3d, K_video, lat_c, lon_c, reprojection_error, pitch_deg, debug, panos_c, use_colour)"]
      M_H_Geolocation_GMAPS_py_228["estimate_pitch(frame_path, K_video, debug)"]
    end
    subgraph M_Q_GPS_processing_py["Q_GPS_processing.py"]
      direction TB
      M_Q_GPS_processing_py_229["android_movie_GPS(video_path, api_key)"]
      M_Q_GPS_processing_py_230["reverse_geocode(lat, lon, api_key)"]
      M_Q_GPS_processing_py_231["csv_to_GPS_dict(csv_path)"]
    end
    subgraph M_Q_Metric_georeferencing_py["Q_Metric_georeferencing.py"]
      direction TB
      M_Q_Metric_georeferencing_py_232["lat_long_to_metres_3dims(coords_list, heights)"]
      M_Q_Metric_georeferencing_py_233["GPS_camerapose_matcher(GPS_dict, cam_pos_dict, max_frame_gap)"]
      M_Q_Metric_georeferencing_py_234["_speed_direction_single(positions_dict, fps)"]
      M_Q_Metric_georeferencing_py_235["speed_direction(positions_dicts, fps)"]
      M_Q_Metric_georeferencing_py_236["graph_real_world(positions_dict)"]
      M_Q_Metric_georeferencing_py_237["metres_to_latlong(positions, lat_0, lon_0, height_0)"]
      M_Q_Metric_georeferencing_py_238["model_to_GPS_calibrated_locations(csv_path_GPS, poses, Source)"]
    end
    subgraph M_Q_subject_analysis_py["Q_subject_analysis.py"]
      direction TB
      M_Q_subject_analysis_py_239["subject_model_positions(recon, masks_dir)"]
      M_Q_subject_analysis_py_240["meeting_calculator(A_real_dict, A_dict_entry, B_real_dict, B_dict_entry, lat_0, lon_0, B_is_camera)"]
      M_Q_subject_analysis_py_241["contact_frames(recon, A_masks_dir, B_masks_dir, touch_px, skip_depthels, max_depthels, depth_tol)"]
      M_Q_subject_analysis_py_242["subject_direction(sub_real_dict)"]
      M_Q_subject_analysis_py_243["find_mask_centroids_in_model_space(subject_recon, masks_dir, RA, analyse)"]
      M_Q_subject_analysis_py_244["subject_to_metric_and_gps_space(subject_recon, masks_dir, lat_0, lon_0, R_mw, s_mw, t_mw)"]
      M_Q_subject_analysis_py_245["rolling_by_frame(frames, meas, window)"]
    end
  end
  subgraph G_Projection___rendering["Projection & rendering"]
    direction LR
    subgraph M_P_projection_mapping_py["P_projection_mapping.py"]
      direction TB
      M_P_projection_mapping_py_246["check_frame_count(preds, frames_dir, name_fmt)"]
      M_P_projection_mapping_py_247["load_full_res_frame(frames_dir, frame_idx)"]
      M_P_projection_mapping_py_248["load_mask(frame_idx, masks_dir)"]
      M_P_projection_mapping_py_249["footprint_size(full_res_shape, lowres_shape)"]
      M_P_projection_mapping_py_250["joint_bilateral_upsample(lowres_depth, lowres_conf, lowres_rgb, full_rgb, sigma_xy, radius, conf_thresh)"]
      M_P_projection_mapping_py_251["reproject(points_world, intrinsic_ref, extrinsic_ref, ortho_params)"]
      M_P_projection_mapping_py_252["scale_intrinsic(intrinsic, sx, sy)"]
      M_P_projection_mapping_py_253["bilinear_sample(img, u, v)"]
      M_P_projection_mapping_py_254["upsample_confidence(lowres_conf, full_shape)"]
      M_P_projection_mapping_py_255["get_full_res_camera(preds, idx, full_shape)"]
      M_P_projection_mapping_py_256["crop_to_mask_region(mask, extra_rgb, depth_low, conf_low, lowres_rgb, intrinsic_full, margin)"]
      M_P_projection_mapping_py_257["unproject_masked(mask, rgb_full, depth_low, conf_low, lowres_rgb, intrinsic_full, extrinsic, margin, conf_thresh)"]
      M_P_projection_mapping_py_258["_as_recon_list(recon)"]
      M_P_projection_mapping_py_259["find_recon_for_frame(recon, frame_idx)"]
      M_P_projection_mapping_py_260["load_frame_inputs(recon, frame_idx, masks_dir)"]
      M_P_projection_mapping_py_261["composite_overlay(recon, main_idx, extra_indices, masks_dir, point_cloud_xforms, depth_margin, splat_radius, new_view, conf_thresh, ortho_params, canvas_size, analyse)"]
      M_P_projection_mapping_py_262["_ortho_view_from_recon(recon, conf_thresh, camera_forward, camera_up, margin_frac, width, height, analyse, resolution)"]
      M_P_projection_mapping_py_263["BEV_tile_render_PM(recon, extra_indices, masks_dir, splat_radius, confidence_threshold, margin_frac, out_path, analyse, resolution)"]
      M_P_projection_mapping_py_264["projection_mapping_sequence(recon, main_idx, extra_indices, masks_dir, point_cloud_xforms, new_view, confidence_threshold, framerate, ffmpeg_path)"]
    end
    subgraph M_P_trace_overlayer_py["P_trace_overlayer.py"]
      direction TB
      M_P_trace_overlayer_py_265["_time_smoothed(frames, positions, fps, window_sec)"]
      M_P_trace_overlayer_py_266["BEV_trace_overlay(recon, extra_indices, new_view, ortho_params, subject_positions, n_frames, bev_png_path, max_width, max_height, camera_trail_color, subject_trail_color, subject_trail_colors, trail_width, trail_opacity, dot_radius, dot_opacity, frustum_length, frustum_half_angle_deg, supersample, fps, smooth_window_sec)"]
    end
    subgraph M_Rendering_R_map_animator_py["Rendering/R_map_animator.py"]
      direction TB
      M_Rendering_R_map_animator_py_267["parse_time(t)"]
      M_Rendering_R_map_animator_py_268["world_xy(lat, lon)"]
      M_Rendering_R_map_animator_py_269["to_image_px(lat, lon, center_lat, center_lon, zoom, img_w, img_h)"]
      M_Rendering_R_map_animator_py_270["auto_zoom(coords, map_size)"]
      M_Rendering_R_map_animator_py_271["fetch_map(center_lat, center_lon, zoom, map_size, scale, maptype, api_key)"]
      M_Rendering_R_map_animator_py_272["path_d(points)"]
      M_Rendering_R_map_animator_py_273["cumulative_lengths(points)"]
      M_Rendering_R_map_animator_py_274["hex_to_rgb(h)"]
      M_Rendering_R_map_animator_py_275["interp_pos(t_frac, time_fracs, px_points)"]
      M_Rendering_R_map_animator_py_276["render_frames(png_bytes, trace_data, map_size, duration, fps, supersample)"]
      M_Rendering_R_map_animator_py_277["export_gif(frames, output, fps)"]
      M_Rendering_R_map_animator_py_278["export_mp4(frames, output, fps)"]
      M_Rendering_R_map_animator_py_279["main(coords, traces, api_key, map_size, map_scale, map_type, zoom, trail_color, trail_width, trail_opacity, dot_radius, dot_opacity, gif, mp4, fps, manual)"]
    end
    subgraph M_Rendering_U_rendering_py["Rendering/U_rendering.py"]
      direction TB
      M_Rendering_U_rendering_py_280["_clean_env(extra)"]
      M_Rendering_U_rendering_py_281["view_ply_sequence_with_scenepic(ply_dir, point_size, pattern, conda_env)"]
      M_Rendering_U_rendering_py_282["_read_ply_points_colors(ply_path)"]
      M_Rendering_U_rendering_py_283["basic_point_cloud_render(ply_dir, model, pattern, width, height, vfov_deg, splat_radius, camera_eye, camera_forward, camera_up, background_color, framerate, ffmpeg_path)"]
      M_Rendering_U_rendering_py_284["_ortho_project_ply_to_cam(ply_path, right, true_up, forward, device)"]
      M_Rendering_U_rendering_py_285["BEV_render(ply_dir, model, pattern, width, height, margin_frac, camera_forward, camera_up, background_color, out_path)"]
    end
  end
  subgraph G_Agent___deliverables["Agent & deliverables"]
    direction LR
    subgraph M_claude_agent_constitution_view_py["claude_agent/constitution_view.py"]
      direction TB
      M_claude_agent_constitution_view_py_286["parse_rules(constitution_path)"]
      M_claude_agent_constitution_view_py_287["print_rules(constitution_path)"]
      M_claude_agent_constitution_view_py_288["constitution_view(excluded_tags, constitution_path)"]
      M_claude_agent_constitution_view_py_289["producer_constitution(constitution_path)"]
    end
    subgraph M_claude_agent_producer_agent_execution_py["claude_agent/producer_agent_execution.py"]
      direction TB
      M_claude_agent_producer_agent_execution_py_290["_make_logger()"]
      M_claude_agent_producer_agent_execution_py_291["_run_producer_agent_async(task_prompt, log)"]
      M_claude_agent_producer_agent_execution_py_292["run_producer_agent(task_prompt, mode)"]
      M_claude_agent_producer_agent_execution_py_293["build_producer_prompt_draft1(brief, analysis_2d_for_decisions, producer_flags)"]
      M_claude_agent_producer_agent_execution_py_294["build_producer_prompt_draft2(brief, analysis_2d_for_decisions, producer_flags)"]
      M_claude_agent_producer_agent_execution_py_295["build_producer_prompt_revision(brief, analysis_2d_for_decisions, producer_flags)"]
    end
    subgraph M_claude_agent_render_paper_edit_py["claude_agent/render_paper_edit.py"]
      direction TB
      M_claude_agent_render_paper_edit_py_296["_esc(s)"]
      M_claude_agent_render_paper_edit_py_297["_toc_row(beat)"]
      M_claude_agent_render_paper_edit_py_298["_beat_block(beat)"]
      M_claude_agent_render_paper_edit_py_299["render_html(data)"]
      M_claude_agent_render_paper_edit_py_300["paper_edit_path(mode)"]
      M_claude_agent_render_paper_edit_py_301["render_and_save(paper_edit_json_path)"]
      M_claude_agent_render_paper_edit_py_302["_beat_flags(beat, fps, gps_signal, errors, notes)"]
      M_claude_agent_render_paper_edit_py_303["derive_flags(paper_edit_json_path, available_flag_keys, fps, gps_signal)"]
      M_claude_agent_render_paper_edit_py_304["derive_and_save_flags(paper_edit_json_path, available_flag_keys, fps, gps_signal)"]
      M_claude_agent_render_paper_edit_py_305["build_tracking_requests(paper_edit_json_path, fps)"]
      M_claude_agent_render_paper_edit_py_306["build_meeting_requests(paper_edit_json_path)"]
      M_claude_agent_render_paper_edit_py_307["_merge_spans(spans)"]
      M_claude_agent_render_paper_edit_py_308["recon_window_frames(paper_edit_json_path, fps, flag)"]
    end
    subgraph M_V_text_summary_py["V_text_summary.py"]
      direction TB
      M_V_text_summary_py_309["gemini_final_report()"]
    end
    subgraph M_W_populate_pptx_py["W_populate_pptx.py"]
      direction TB
      M_W_populate_pptx_py_310["set_text(shape, value)"]
      M_W_populate_pptx_py_311["substitute_placeholders(shape, report)"]
      M_W_populate_pptx_py_312["replace_image(slide, shape, image_path)"]
      M_W_populate_pptx_py_313["replace_video(slide, shape, video_path)"]
      M_W_populate_pptx_py_314["_col_letter(n)"]
      M_W_populate_pptx_py_315["_update_range_end_row(ref, end_row)"]
      M_W_populate_pptx_py_316["_populate_num_cache(cache_el, ns, values)"]
      M_W_populate_pptx_py_317["_populate_str_cache(cache_el, ns, value)"]
      M_W_populate_pptx_py_318["update_scatter(shape, csv_path)"]
      M_W_populate_pptx_py_319["dispatch_chart(shape)"]
      M_W_populate_pptx_py_320["add_hyperlink(shape, target_path, output_dir)"]
      M_W_populate_pptx_py_321["replace_video_thumb(slide, shape, image_path)"]
      M_W_populate_pptx_py_322["delete_slide(prs, slide)"]
      M_W_populate_pptx_py_323["_load_csv_dict(path)"]
      M_W_populate_pptx_py_324["_renumber_person_keys(report)"]
      M_W_populate_pptx_py_325["populate(report_template)"]
    end
  end
```
