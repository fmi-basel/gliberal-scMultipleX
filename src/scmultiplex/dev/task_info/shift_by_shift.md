### Purpose
- Copy and shift 2D or 3D label image from reference round to moving round(s)
- Shift using pre-calculated x,y,z translations, e.g. from Calculate Linking Consensus task
- Allows user to propogate segmentation run only on reference round to subsequent rounds, using rough registration
- Useful for copying over organoid-level segmentation prior to more high-res registration on single-cell level

### Outputs
- A new **label image** saved as a label in moving rounds with '_shifted' suffix
- A new corresponding **masking ROI table** saved as `{new_label_name}_ROI_table` in the moving rounds

### Limitations
- Applies x,y,z translation only
- Translation must have been saved in a ROI table with columns "translation_z", "translation_y", "translation_x"
