### Purpose
- **Measures pixel features** from existing segmentation images in an OME-Zarr dataset.
- Measure number of pixels above given intensity value for each label object in image. User sets thresholds per channel in task input.
- Compound task that can be run over single or multiplexing rounds.

### Outputs
- Feature table that includes the pixel counts per channel, centroid, and number of total pixels for each measured object in image.

### Limitations
- Classifies pixels as positive only based on intensity threshold, no classifier or combined features currently supported.
