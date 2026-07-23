# Conventions

## Asset paths in CSV reports

When a pipeline stage saves an output file (image, video, etc.) under `assets_dir`,
save it with an absolute path built from `Path(assets_dir) / f"{asset_name}_<suffix>"`,
but log the path passed to `add_to_report(...)` as a path relative to `assets_dir`
(`str(outpath.relative_to(assets_dir))`), not the absolute path.

Example: `src/C_video_editor.py:137-149`.
