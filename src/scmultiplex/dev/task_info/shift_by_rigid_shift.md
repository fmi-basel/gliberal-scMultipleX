### Purpose
- Copy and shift 2D or 3D label image from reference round to moving round(s)
- Shift using on the fly rigid transformation of the linked 2D MIP image
- Allows user to propogate segmentation run only on reference round to subsequent rounds, using rough registration
- Useful for copying over organoid-level segmentation prior to more high-res registration on single-cell level
- Can be more accurate than shift_by_shift task since takes into account rotation.

### Outputs
- A new **label image** saved as a label in moving rounds with '_shifted' suffix
- A new corresponding **masking ROI table** saved as `{new_label_name}_ROI_table` in the moving rounds

### Limitations
- Applies x,y,z translation and rotation (rigid transformation; no scaling or nonrigid warping)
- Objects in 2D MIP image must be consensus linked, so that there is equal object number with matching IDs between rounds.
