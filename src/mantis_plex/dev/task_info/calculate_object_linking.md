### Purpose
- Links segmented objects between a reference and an alignment acquisition within a single well of an HCS OME-Zarr dataset.
- Calculates object shifts using segmentation label maps, aligns objects, and identifies matching labels based on an Intersection over Union (IoU) cutoff threshold.
- Generates a linking table that maps object labels from the reference acquisition to those in the alignment acquisition.

### Outputs
- A linking table stored in the alignment acquisition directory.
- The table includes matched object pairs and their IoU scores.

### Limitations
- Only works for HCS OME-Zarr datasets where a **single well ROI** is used for linking. Multi-ROI processing (e.g., for FOV ROI tables) is not yet supported.
- Requires segmentation label maps to be provided for both the reference and alignment acquisitions.
- Matching is performed using an IoU threshold; objects below the threshold are not linked.
- Pixel sizes must match between the reference and alignment acquisitions for accurate registration.
