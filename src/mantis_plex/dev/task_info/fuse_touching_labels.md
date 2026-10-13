### Purpose
- Fuse touching labels in segmentation images, in 2D or 3D. Connected components are identified during labeling
    based on the connectivity argument. For a more detailed explanation of 1- or 2- connectivity, see documentation
    of skimage.measure.label() function. When set to None (default), full connectivity (ndim of input array) is used.

- Input is segmentation image with 0 value for background. Anything above 0 is assumed to be a labeled object.
    Touching labels are labeled in numerically increasing order starting from 1 to n, where n is the number of
    connected components (objects) identified.

- This task has been tested for fusion of 2D MIP segmentation. Since fusion must occur on the full well numpy
    array loaded into memory, performance may be poor for large 3D arrays.

### Outputs
- The fused label image is saved as a new label in zarr, with name {label_name_to_fuse}_fused.
- The new ROI table for the fused label image is saved as a masking ROI table, with name
    {label_name_to_fuse}_fused_ROI_table.
