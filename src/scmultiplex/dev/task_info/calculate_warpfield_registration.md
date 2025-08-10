### Purpose
- Calculate warpfield warp map between reference/moving object pairs in multiplexing rounds
- GPU-based 3D non-rigid volumetric registration
- See https://github.com/danionella/warpfield/tree/main

### Outputs
- Warmap for each moving object, saved in 'registration' folder as .npz file

### Limitations
- Uses masking ROI tables to load object pairs between reference/moving round
- Assumes objects have been linked across rounds to have same label id and shape (no padding supported)
