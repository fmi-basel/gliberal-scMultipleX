## Purpose

- Copy ROI tables from the reference multiplexing round to all submitted moving rounds.
- Automatically copy the corresponding label images for any masking ROI tables.
- Preserve the original label image data type when creating copied labels.
- Skip the reference round; only moving rounds are modified.
- Existing labels and tables are overwritten only if `overwrite=True`.


## Inputs

- **roi_table_names_to_copy_from_ref**: List of ROI table names to copy from the reference round. For tables of type `masking_roi_table`, the corresponding label images are copied automatically.
- **overwrite** *(default: `False`)*: If `True`, overwrite existing tables and labels in the moving rounds.

## Outputs

- Copies the requested ROI tables from the reference round to each moving round.
- Copies any label images referenced by masking ROI tables to the corresponding moving rounds.
- Builds label pyramids for copied label images.

## Limitations

- Assumes that reference and moving images have compatible spatial metadata (shape, pixel size, axes).
- Only copies labels and tables; in future consider supporting copying of specific subfolders, e.g. meshes or registration, if needed.
