![Asset 13scMultiplex](https://user-images.githubusercontent.com/25291742/227270190-34aeb814-37e4-49ef-8347-dccab94684c8.png)

# Overview
scMultipleX is a software package for feature extraction of microscopy imaging data. It provides workflows for feature extraction of segmentated objects (e.g. organoids) and single cells, and for linking of objects and cells over multiplexing rounds. It supports 2D and 3D imaging data, and single-round or multiplexed experiments. scMultipleX uses [Prefect](https://docs.prefect.io/) (v1.4) for parallelized processing, and assumes input data pre-proprecessed with [Drogon](https://github.com/fmi-basel/job-system-workflows).

The workflow consists of the following tasks:
- **Task 0 Build Experiment:** Initialize output data storage structure with [FAIM-HCS](https://github.com/fmi-faim/faim-hcs) (v0.1.1)
- **Task 1 Feature Extraction:** Perform 2D object-level and 3D single-cell-level feature extraction and nuclear to membrane linking
- **Task 2 Organoid Multiplex:** Link objects across multiplexing rounds
- **Task 3 Nuclear Multiplex:** Link nuclei within objects across multiplexing rounds
- **Task 4 Aggregate Features:** Output measured features for each round and objects type (e.g. organoids, nuclei, membranes)
- **Task 5 Combine Nuclear and Membrane Features:** Output combined nuclear and membrane features based on nuclear to membrane linking
- **Task 6 Aggregate Organoid Multiplex:** Output measured object features across multiplexing rounds
- **Task 7 Aggregate Nuclear Multiplex:** Output measured nuclear features across multiplexing rounds

## Cite
The nuclei linking is built on [PlatyMatch](https://github.com/juglab/PlatyMatch) by Manan Lalit [DOI](https://doi.org/10.1007/978-3-030-66415-2_30).
