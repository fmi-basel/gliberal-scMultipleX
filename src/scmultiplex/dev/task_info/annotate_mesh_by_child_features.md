### Purpose
- Annotate parent mesh (.stl) vertices (points) by child features
- This allows visualization of single-cell features on tissue-scale mesh
- For visualization in Paraview

### Outputs
- Saves .vtp mesh with embedded point annotations.

### Limitations
- Assumes feature extraction table and mesh have same physical unit spacing and scaling.
- Each point of mesh is assigned to closest child object by euclidean distance to centroid.
