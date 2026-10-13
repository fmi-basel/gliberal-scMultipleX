<img width="4616" height="2094" alt="Asset 16@600x" src="https://github.com/user-attachments/assets/d5f04772-56e3-4b94-9758-04306e425144" />


`Mantis` is an ecosystem of modular tools for image processing and quantitative analysis of high-content, multiplexed, and volumetric microscopy data. The name is inspired by the mantis shrimp and its complex visual system, which detects light across spectral channels 🦐🌈

`Mantis-plex` provides tools for processing multiplexed imaging datasets, including image registration, object linking, and z-illumination correction across rounds. Its functions are wrapped as [Fractal](https://fractal-analytics-platform.github.io/) tasks operating on the [OME-Zarr](https://ngff.openmicroscopy.org/) image format, enabling reproducible image-processing workflows that can be executed on compute clusters.


#### 🚧 Mantis is currently evolving!

Mantis-plex is being reorganized into four complementary packages: Mantis-plex, Mantis-quant, Mantis-surf, and Mantis-seg, each providing modular analysis tools organized by functionality. 

<img width="4893" height="2245" alt="Asset 2@600x" src="https://github.com/user-attachments/assets/0fa294b0-90d4-4990-b046-5bbe91dbb54e" />

More information about Mantis-quant, Mantis-surf, and Mantis-seg is coming soon!

## Installation

### From source

Clone the repository and install Mantis-plex:

```bash
git clone https://github.com/fmi-basel/Mantis-plex.git
cd Mantis-plex
pip install .
```

To include optional dependencies for spherical harmonics and warpfield registration:

```bash
pip install ".[spherical-harmonics,warpfield]"
```

## Fractal workflow steps

`Mantis-plex` functions are implemented as Fractal tasks located under:

```text
src/mantis_plex/fractal/
```

These tasks can be collected and executed within a Fractal deployment alongside tasks from the Fractal ecosystem.


<img width="4094" height="1890" alt="Asset 4@600x" src="https://github.com/user-attachments/assets/1eb2ba77-886f-422b-840c-b388d3040998" />


## License

Unless otherwise stated in each individual module, all Mantis-plex components are released under the BSD 3-Clause License, and copyright is held by the Friedrich Miescher Institute for Biomedical Research.

## Contributors

Developed in the Liberali Lab at the Friedrich Miescher Institute for Biomedical Research (FMI), Basel, Switzerland by [@nrepina](https://github.com/nrepina), with development support from [@jluethi](https://github.com/jluethi), [@lorenzocerrone](https://github.com/lorenzocerrone), [@enricotagliavini](https://github.com/enricotagliavini), and [@tibuch](https://github.com/tibuch).

Point-cloud based multiplexed linking is built on [PlatyMatch](https://github.com/juglab/PlatyMatch) ([DOI](https://doi.org/10.1007/978-3-030-66415-2_30)) by Manan Lalit.

Warpfield registration is based on [Warpfield](https://github.com/danionella/warpfield/tree/main) ([DOI](https://doi.org/10.1038/s41467-023-43741-x)).

Spherical harmonic computation is based on [aisc-shparam](https://github.com/AllenCell/aics-shparam).
