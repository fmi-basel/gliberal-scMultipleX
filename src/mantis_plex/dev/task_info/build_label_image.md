### Purpose
- Clean up incomplete label image on disk
- Build pyramid structure on disk for label image
- Make masking ROI table based on label image
- This is useful if a segmentation task was run but failed with OoM error at pyramid building step.
- Works on both 2D and 3D zarr arrays

### Outputs
- Pyramid structure for label image (optional), matching chunking and number of levels of parent zarr array
- Masking ROI table (optional)

### Limitations
- Only works for label images, not image arrays
