## Purpose

- Computes a 3D non-rigid volumetric registration between linked reference and moving ROIs using the GPU-based Warpfield algorithm.
- Performs registration independently for each ROI defined in the selected ROI table.
- Saves the resulting warp map for each ROI to disk for later application with the `apply_warpfield_registration` task.
- Optionally modify the Warpfield recipe by specifying a custom recipe file or overriding the pre-filter clipping threshold.
- See Warpfield documentation for more information: https://github.com/danionella/warpfield/tree/main

## Inputs

- **registration_channel**: Image channel used to calculate the warpfield registration. The channel must be present in both the reference and moving images.
- **roi_table_name**: Name of the generic ROI table used to load corresponding reference and moving ROIs. ROI IDs must match between rounds.
- **path_to_warpfield_recipe** *(optional)*: Path to a custom Warpfield recipe (`.yml`). If omitted, the default Warpfield recipe is used.
- **warpfield_pre_filter_clip_thresh** *(optional)*: Override for the pre-filter clipping threshold defined in the Warpfield recipe; this removes background signal.
- **registration_name** *(default: `warpfield`)*: Name of the output registration folder created under `registration/` in OME-Zarr structure.
- **overwrite_folder** *(default: `False`)*: If `True`, overwrites the existing registration folder before saving new warp maps.


## Outputs

- One compressed Warpfield map (`.npz`) per successfully registered ROI, saved in:

  ```text
  registration/<registration_name>/<label>.npz
  ```

- Each warp map stores:
  - moving ROI shape,
  - reference ROI shape,
  - block size,
  - block stride,
  - dense warp field.

### Limitations

- Assumes reference and moving ROIs have been linked across rounds to have same label id and general shape. Slight mismatches in shape (recommended not more than a few pixels) are supported with padding.
- Registration is performed independently for each ROI; no continuity is enforced across ROI boundaries.
- Missing moving ROIs are skipped with a warning.
- Depending on Warpmap recipe and ROI sizes, this task can require significant GPU memory.
