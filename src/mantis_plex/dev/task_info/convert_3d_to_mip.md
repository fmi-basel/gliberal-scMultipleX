### Purpose
- Convert 3D segmentations into 2D maximum intensity projection (MIP) segmentations
- Creates MIP along z axis
- Useful for generating 2D MIP for multiplexed registration

### Outputs
- A new **MIP label image** saved as a label in the `_mip` 2D zarr.
- A new **masking ROI table** saved as `{new_label_name}_ROI_table` in the 2D zarr.

### Limitations
- If multiple labels overlap along z, they are collapsed into a single value corresponding to the higher label id number (max) of the labels.
