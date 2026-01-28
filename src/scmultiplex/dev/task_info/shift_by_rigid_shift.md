### Purpose
- Copy and shift 2D or 3D label image from reference round to moving round(s) using rigid transform (EuclideanTransform) calculated from 2D label images
- Transform (rotation + translation) label using on the fly 2D (xy) rigid transformation of the linked 2D MIP label image
- Same XY transform applied across all Z for 3D labels
- 2D label image must be linked across rounds to have same number and matching ID's of objects
- Allows user to propogate segmentation run only on reference round to subsequent rounds, using rigid registration
- Useful for copying over organoid-level segmentation prior to more high-res registration on single-cell level
- Can be more accurate than shift_by_shift task since takes into account rotation.
- Adds 0 padding when chunks shifted beyond image canvas edge.

### Outputs
- A new **label image** saved as a label in moving rounds with '_shifted' suffix
- A new corresponding **masking ROI table** saved as `{new_label_name}_ROI_table` in the moving rounds
- Saves 2D rigid transform in 'registration' folder of moving image following NGFF coordinateTransformations specification: JSON file of sequence (rotation, translation) for forward transform from moving to reference image in physical units (y,x).

### Limitations
- Applies x,y translation and rotation only (rigid Euclidean transformation only; no scaling or warping)
- No z transformation: assumes same x,y translations applied across z
- Objects in 2D MIP image must be consensus linked, so that there is equal object number with matching IDs between rounds.
