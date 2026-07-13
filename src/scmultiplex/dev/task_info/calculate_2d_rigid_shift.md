### Purpose

* Calculate a 2D rigid registration between a 2D reference-round label image and a 2D moving-round label image.
* Estimate a Euclidean transformation consisting of XY rotation and translation.
* Use label image object centroids as registration landmarks.
* Require label images to be linked, meaning that corresponding objects have identical label IDs between the reference and moving rounds. E.g. organoid #3 in reference is organoid #3 in moving round, even if it can have different segmentation shape and centroid.
* Save the resulting transformation in the moving-round OME-Zarr image for use by downstream label-transformation tasks such as `shift_by_rigid_shift`.

### Inputs

* `zarr_url`: Path or URL of the moving-round OME-Zarr image.
* `init_args`: Registration initialization arguments supplied by the Fractal registration initialization task. These contain the reference-round OME-Zarr URL.
* `label_name_for_2D_rigid_transform`: Name of the linked 2D label image present in both the reference and moving OME-Zarr images.
* `registration_name`: Name of the registration output folder. Defaults to `rigid_2D`.
* `overwrite_folder`: If `True`, remove an existing registration output folder with the same name before saving the newly calculated transform. If `False`, the task raises an error when the output folder already exists.


### Outputs

* A registration folder created inside the moving-round OME-Zarr image:

  ```text
  {zarr_url}/registration/{registration_name}/
  ```

* Containing a rigid-transformation file named:

  ```text
  sequence.json
  ```

* The saved transformation contains the rotation matrix and translation offset in physical units (e.g. um) that map the moving image onto the reference image.

* The .JSON file structure corresponds to the proposed Coordinate Systems and Transformations NGFF spec: https://ngff.openmicroscopy.org/rfc/5/

* The saved transformation can be loaded and converted back to pixel coordinates by downstream tasks before being applied with `scipy.ndimage.affine_transform`.

### Note

The transformation is initially estimated from the reference centroids to the moving centroids:

  ```text
reference → moving
  ```
Before saving, it is converted to the inverse-coordinate convention required by scipy.ndimage.affine_transform:
  ```text
reference input coordinate = matrix @ moving output coordinate + offset
  ```
Therefore, sequence.json stores the reverse/inverse coordinate mapping from the moving output grid to the reference input image, expressed in physical (y, x) units.

This transform can be loaded directly by a downstream task and supplied to scipy.ndimage.affine_transform to resample the reference label into the moving image coordinate system.

### Limitations

* Both reference and moving labels must be two-dimensional images represented as arrays with shape `(1, Y, X)`.
* Supports 2D Euclidean registration only: XY rotation and translation.
* Does not support scaling, shear, nonlinear deformation, local warping, or Z-axis transformation.
* Does not support true 3D label volumes or on-the-fly 2D projection.
* The reference and moving labels must contain exactly the same set of label IDs.
* Uses all detected object centroids in a least-squares transform estimate and does not perform outlier rejection.
* Incorrectly linked objects or inaccurate segmentations can therefore strongly affect the estimated transform.
* Requires enough (more than 3) distinct corresponding objects to determine a Euclidean transform.
* Label value `0` is treated as background
