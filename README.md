![Asset 14scMultiplex](https://user-images.githubusercontent.com/25291742/227540510-b00e0b8e-241e-492f-9b67-0d01bb4250e2.png)

# scMultipleX

`scMultipleX` provides functions for image processing workflows for high-content, multiplexed, and volumetric microscopy datasets.

The package supports workflows for segmentation post-processing, object linking across multiplexing rounds, registration, intensity correction, feature extraction, mesh generation, and 3D shape analysis.

scMultipleX functions are wrapped as [Fractal](https://fractal-analytics-platform.github.io/) tasks based on the OME-Zarr image format.
scMultipleX tasks enable users to build reproducible image-processing workflows from modular tasks and execute them locally or on compute clusters.

The tasks provided by scMultipleX are located under:

```text
src/scmultiplex/fractal/
```

and can be collected and executed within a Fractal deployment alongside tasks from the Fractal ecosystem.


## Main Functionality

### Segmentation and Label Processing

Tasks for preparing and cleaning label images:

- Build label images
- Expand labels
- Fuse touching labels
- Clean up 3D cell segmentations
- Segment images by intensity threshold
- Convert 3D images to maximum-intensity projections

### Object Linking and Relabeling

Tasks for linking related objects across segmentations or imaging rounds:

- Calculate object linking
- Calculate linking consensus
- Relabel objects using linking consensus
- Link parent and child objects, such as nuclei within cells or organoids

### Registration and Multiplexing

Tasks for aligning multiplexed imaging rounds:

- Calculate Warpfield registration for 3D pixel-based registration
- Apply Warpfield registration
- Calculate and shift images by rigid transformation (translation and rotation)
- Post-registration cleanup
- Detect clipped ROIs across rounds

### Illumination Correction

Tasks for correcting Z-dependent illumination effects:

- Calculate z-illumination correction
- Apply z-illumination correction

### Feature Measurements

Tasks for extracting quantitative measurements from segmented objects:

- Morphology measurements
- Intensity measurements
- Pixel-threshold measurements
- Measurements within masked parent ROIs
- Regionprops-based 2D and 3D feature tables

### Mesh and 3D Shape Analysis

Tasks for generating and analyzing 3D surface meshes:

- Generate multiscale surface meshes
- Measure mesh-based features
- Calculate spherical harmonics from label images
- Annotate child meshes
- Annotate meshes using child-object features

### Fractal Workflow Initialization

Helper tasks for selecting images and defining parallelization lists:

- Select single imaging rounds
- Select multiple imaging rounds
- Select multiplexing pairs
- Select reference rounds
- Select illumination-correction rounds

## Installation

### From source

```bash
git clone https://github.com/fmi-basel/gliberal-scMultipleX.git
cd gliberal-scMultipleX
pip install -e .
```

### With dependencies

```bash
pip install -e ".[spherical-harmonics, warpfield]"
```

## License

BSD 3-Clause License.

## Contributors

Developed in the Liberali Lab at the Friedrich Miescher Institute for Biomedical Research (FMI), Basel, Switzerland by [@nrepina](https://github.com/nrepina), with development support from [@jluethi](https://github.com/jluethi), [@lorenzocerrone](https://github.com/lorenzocerrone), [@enricotagliavini](https://github.com/enricotagliavini), and [@tibuch](https://github.com/tibuch).

Unless otherwise stated in each individual module, all scMultipleX components are released according to a BSD 3-Clause License, and Copyright is with Friedrich Miescher Institute for Biomedical Research.

Point-cloud based multiplexed linking is built on [PlatyMatch](https://github.com/juglab/PlatyMatch) ([DOI](https://doi.org/10.1007/978-3-030-66415-2_30)) by Manan Lalit.

Warpfield registration is based on [Warpfield](https://github.com/danionella/warpfield/tree/main) ([DOI](https://doi.org/10.1038/s41467-023-43741-x)).

Spherical harmonic computation is based on [aisc-shparam](https://github.com/AllenCell/aics-shparam).
