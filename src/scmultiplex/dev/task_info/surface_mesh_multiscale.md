### Purpose
- Calculates **3D surface meshes** for parent objects (e.g., tissues, organoids) based on 3D child-level segmentation (e.g., nuclei).
- Optionally applies **multiscale label fusion** to estimate a smooth parent shape by merging child objects.
- Generates smoothened surface meshes using **VTK algorithms**, with optional mesh decimation for reduced complexity.
- Outputs 3D meshes in `.stl` or `.vtp` format and a new well label map in the dataset.

### Outputs
- **Surface meshes** of parent objects, saved as `.stl` (single object) or `.vtp` (multi-object) files in the dataset’s `meshes` folder.
- A **new label map** containing fused child-level objects, saved in the OME-Zarr dataset (only if multiscale processing is enabled).
- A **bounding-box ROI table** corresponding to the new label map.

### Limitations
- Requires pre-segmented child objects and a parent object ROI table.
- Multiscale processing requires a **parent label** for accurate object grouping and fusion.
- Label map outputs may have **overlaps clipped**, where higher-label IDs take precedence in dense regions.
- Mesh quality can vary with complex geometries; manual tuning of smoothing parameters may be needed for optimal results.
