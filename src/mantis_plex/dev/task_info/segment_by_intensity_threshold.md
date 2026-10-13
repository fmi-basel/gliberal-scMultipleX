### Purpose
- Performs **full 3D object segmentation** of raw intensity images using intensity thresholding.
- Combines two intensity channels, applies **Gaussian smoothing** and **Canny edge detection** for refined masks.
- Filters out debris and neighboring objects by selecting the **largest connected component** within a masked region.
- Outputs a new 3D segmentation label image and an updated masking ROI table.

### Outputs
- A **new 3D label image** stored in the dataset, with refined object segmentation.
- A corresponding **bounding-box ROI table** saved as `{output_label_name}_ROI_table`.

### Limitations
- Requires pre-segmented 2D MIP-based ROI regions as input for masking.
- Supports intensity thresholding with either **Otsu's method** or a user-defined threshold.
- Assumes consistent image resolution and pixel intensities across channels.
- Regions with extreme intensity variations or overlapping objects may require manual parameter tuning for optimal results.
