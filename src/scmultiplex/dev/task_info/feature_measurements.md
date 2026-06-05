### Purpose
- **Measures intensity and morphology features** from existing segmentation images in an OME-Zarr dataset.
- Runs on both 2D and 3D images.
- Optionally computes advanced morphological metrics (see below)
- Supports both intensity-based and morphology-only measurements:
   - If no input intensity channels are provided, the task calculates morphology features only.
   - For intensity measurements, user specifies channels in image to be used.
- Enables **measurements within masked objects** (e.g., measuring nuclei properties within organoids) by specifying an `input_ROI_table` that defines parent regions, such as organoid ROIs.

### Outputs

- Feature table with the following measurements:

  - **Spatial coordinates**
    - `x_pos`, `y_pos`, `z_pos` — Object centroid coordinates in physical units.
    - `z_pos_pix` — Object centroid position in z expressed in pixel coordinates.

  - **Morphology**
    - `is_touching_border_xy` — Whether the object touches the image border in x/y. Image borders are defined by the ROI used for iteration, as specified by `input_roi_table_name`.. For example if well_ROI_table, border is the well edge. If it is a masking ROI table, border is edge of each masking object.
    - `imgdim_x`, `imgdim_y`, `imgdim_z` — Dimensions in pixels of the analyzed image or ROI. `imgdim_z` given only in 3D case.
    - `area_bbox` — Area/volume of the object's bounding box.
    - `area_convhull` — Area/volume of the object's convex hull.
    - `equivDiam` — Equivalent diameter of a sphere/circle with the same volume/area.
    - `extent` — Fraction of the bounding box occupied by the object.
    - `solidity` — Fraction of the convex hull occupied by the object.
    - `axis_major_length` — Length of the major axis of the object.
    - `axis_minor_length` — Length of the minor axis of the object.
    - `minmajAxisRatio` — Ratio of minor to major axis length (min divided by maj).
    - `aspectRatio_equivalentDiameter` — Ratio of major axis length to equivalent diameter.
    - **2D-only features:**
      - `area` - Object area in physical units (e.g. um^2) (2D only).
      - `perimeter` — Object perimeter in physical units (2D only).
      - `concavity` — Fraction of the convex hull area not occupied by the object (2D only).
      - `asymmetry` — Normalized distance between the centroid of the object and the centroid of its convex hull. This centroid distance is divided by the square root of the object area, making the metric approximately independent of object size. Values close to 0 indicate a more symmetric shape, while larger values indicate increasing asymmetry or irregularity (2D only).
      - `eccentricity` — Elongation metric based on an ellipse fitted to the object shape. Values range from 0 to 1, where 0 corresponds to a perfect circle and values approaching 1 correspond to increasingly elongated objects (2D only).
      - `circularity` — Shape compactness metric calculated from the object's area and perimeter (`4π × area / perimeter²`). A perfect circle has a value of 1, while objects with elongated outlines, protrusions, indentations, or irregular boundaries have progressively lower values (2D only).
      - `disconnected_components` — Whether the object mask contains disconnected components (2D only).
    - **3D-only features:**
      - `volume` — Object volume in physical units (e.g. um^3) (3D only).
      - `surface_area` — Object surface area estimated using a marching-cubes mesh (3D only).
      - `is_touching_border_z` — Whether the object touches the image border in z (3D only).

  - **Per-channel intensity statistics** (for each requested input channel)
    - `intensity_mean` — Mean intensity within the segmented object.
    - `intensity_max` — Maximum intensity.
    - `intensity_min` — Minimum intensity.
    - `percentile25`, `percentile50`, `percentile75`, `percentile90`, `percentile95`, `percentile99` — Intensity percentiles.
    - `stdev` — Standard deviation of intensity values.
    - `skew` — Skewness of the intensity distribution.
    - `kurtosis` — Kurtosis of the intensity distribution.

  - **Intensity-weighted centroid measurements** (for each requested input channel)
    - `x_pos_weighted`, `y_pos_weighted`, `z_pos_weighted` — Intensity-weighted centroid coordinates.
    - `x_massDisp`, `y_massDisp`, `z_massDisp` — Displacement between the geometric centroid and the intensity-weighted centroid, indicating asymmetry in signal distribution.

- Units: all features are in physical units from the OME-Zarr pixel metadata (e.g. um) unless otherwise noted above. Anisotropic voxel spacing is supported.
- If Measure Morphology is set to True, all morphological features are measured. If False, only `Spatial Coordinates` and `Intensity` metrics are measured.
- Channel-specific measurements are prefixed with the channel identifier (e.g. `C01.intensity_mean`, `C01.x_pos_weighted`).
### Limitations
- Does not support measurements at lower resolutions (e.g., beyond level 0).
