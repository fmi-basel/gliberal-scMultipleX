### Purpose
- Calculate z-illumination correction curve for each 3D object in well. Task assumes 3D object segmentation has been
    performed to identify regions of interest. Task also assumes objects were imaged with a uniform
    stain whose intensity drop-off along z-axis is representative of z-correction that needs to be applied.
- Task has been tested with input object segmentation (specified with "label_name" and "roi_table") performed in
    3D, where the z-extent of the objects has also been roughly defined. The task should also perform on 2D-based
    segmentation in 3D, but can be sensitive to debris axially above or below the object. This scenario has not been
    thoroughly tested. In the ideal case, the 3D segmentation should tightly fit the object and exclude any lumen or
    empty space; in this way, z-correction is calculated using signal from the object itself, not background or
    debris.
- Processing proceeds as follows, with masked loading of each segmented object:
- (1) The loaded raw channel image is masked by the segmentation image. For each z-slice, the intensity value at the
    user-specified percentile ("percentile" input) is computed. This generates the intensity vs. z curve, which is
    normalized to a maximum value of 1 (i.e. range 0 to 1) and used for subsequent analysis steps. This intensity
    curve measures the drop-off in intensity over z.
- (2) To identify the start and end of the object in z, it is assumed that the intensity is low in regions outside
of the organoid. The smoothened first derivative of the intensity curve is computed and used to identify peaks
at the start and end, which would correspond to regions of large changes in intensity (from low to high for the
start peak, and from high to low for the end peak). These start and end values are used to crop the intensity
curve, to use only the region of the object for model fitting.
- (3) The cropped intensity curve (intensity vs. z) is used to fit 3 models: "Polynomial", "ExponentialDecayConcave",
"ExponentialDecayConvex". The best fit is chosen by the highest chi-sqr value. The model is used to evaluate
the intensity at each z, which effectively smoothens the decay curve. These values are again normalized to a max
value of 1. For z-slices below the identified object region, the first correction value is repeated until the first
z plane. For z-slices above the identified object region, the last correction value is repeated until the end of
the z-stack. For quality control, plots of the intensity curve and fits are saved for each object in a
'plots' folder within the ome-zarr structure.
- (4) The identified correction values (0 to 1, where 1 is no correction, and lower values indicate stronger
decay and thus stronger correction) are saved as an anndata table within the image. Rows of the adata table
correspond to the object label that the curve was calculated from. Columns correspond to z-slice, where column 0 is
the first (typically lowest) z-slice of the object. This table is to be used as input for the
Apply Z-illumination Correction Task, where channel images are divided by the correction values.

### Outputs
- Output is anndata table of correction values, with dimensions object x z-slice (row x col). It contains an obs
    column 'label' which stores the object label as string. The table is saved per channel, with name
    "zillum_{channel}", where channel is the wavelength id or label of the channel used to calculate the
    correction values.
- For quality control, plots of the intensity curve and fits are saved for each object in a
   'plots' folder within the ome-zarr structure. Within the plots folder, each channel has a separate folder specified
   by {channel}_{timestamp}. In this way, if the task is rerun, the tables are overwritten but the plots of older runs
   are assigned a new timestamp so that outputs using different parameters can be compared.
