### Purpose
- Apply warp map (output of calculate_warpfield_registration) to all channels of moving image.

### Outputs
- New registered 3D OME-Zarr image that is aligned to reference round
- This image has same dimensions and placement of linked object regions as reference round
- Label images and ROI tables are optionally copied to this new image

### Limitations
- Uses masking ROI tables to load object pairs between reference/moving round
- Assumes objects have been linked across rounds to have same label id and shape (no padding supported)
