### Purpose
- Annotate grouped child mesh (.vtp) vertices (points) by child features
- This allows visualization of single-cell features on cell-scale mesh
- For visualization in Paraview

### Outputs
- Saves .vtp mesh with embedded point annotations.

### Limitations
- Assumes grouped child mesh (.vtp) generate by Surface Mesh Multiscale task and contains "label_id" annotations to identify each child.
