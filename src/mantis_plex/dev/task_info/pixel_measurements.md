### Purpose
- **Measures pixel features** from existing segmentation images in an OME-Zarr dataset.
- Measure number of pixels above given intensity value for each label object in image. User sets thresholds per channel in task input.
- Compound task that is run on single image.

### Outputs
- A feature table containing, for each labeled object:
  - Number of pixels above the intensity threshold.
  - Sum of the intensities of pixels above the intensity threshold.
  - Object centroid coordinates.
  - Total number of pixels in the object.

### Limitations
- Classifies pixels as positive only based on intensity threshold, no classifier or combined features currently supported.
