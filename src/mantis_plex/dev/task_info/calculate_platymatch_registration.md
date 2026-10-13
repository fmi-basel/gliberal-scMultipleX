### Purpose
- Calculates point-cloud-based registration between segmentation images using **PlatyMatch**.
- Works well for **complex 3D registration**.
- Aligns sub-objects (e.g., nuclei) that belong to parent objects (e.g., organoids) by calculating **affine** and optionally **free-form deformation** transformations.
- Outputs linking tables of matched sub-objects and optionally saves transformation matrices to disk.

### Outputs
- A **linking table** that maps sub-objects between reference and alignment rounds using affine and/or free-form deformation (FFD) transformations.
- Transformation matrices (optional), saved on disk for each object pair.

### Limitations
- Only supports **single well ROI tables**; multi-ROI processing (e.g., FOV ROIs) is not yet implemented.
- Requires parent objects to be linked in a prior step using a **consensus linking table**.
- Assumes consistent pixel sizes between reference and alignment rounds for accurate registration.
- Relies on sufficient sub-object counts for alignment; regions with fewer than 3 sub-objects are skipped.
