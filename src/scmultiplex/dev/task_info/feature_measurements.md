### Purpose
- **Measures intensity and morphology features** from existing segmentation images in an OME-Zarr dataset.
- Computes advanced 3D morphology metrics, including surface area, using extended `regionprops` measurements.
- Supports both intensity-based and morphology-only measurements:
   - If no input intensity channels are provided, the task calculates morphology features only.
   - For intensity measurements, channels can be specified individually, allowing flexibility across different image inputs.
- Enables **measurements within masked objects** (e.g., measuring nuclei properties within organoids) by specifying an `input_ROI_table` that defines parent regions, such as organoid ROIs.

### Limitations
- Currently tested only on image data in the **CZYX** format.
- Measurement accuracy and performance may depend on the spacing and resolution of input images.
- Does not support measurements at lower resolutions (e.g., beyond level 0).
