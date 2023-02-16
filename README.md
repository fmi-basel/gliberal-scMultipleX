# gliberal-scMultipleX
scMultipleX provides workflows for feature extraction of imaging data. It supports 2D and 3D imaging data, and single-round or multiplexed experiments.

The workflow consists of:
- Organization into FAIM-HCS (v0.1.1) folder structure
- Feature extraction 
- Nuclear to membrane linking
- Organoid linking across multiplexing rounds 
- Single-cell linking across multiplexing rounds 
- Aggregation of experiment data into output csv files

## Cite
The nuclei linking is built on [PlatyMatch](https://github.com/juglab/PlatyMatch) by Manan Lalit.
```text
@InProceedings{10.1007/978-3-030-66415-2_30,
author="Lalit, Manan and Handberg-Thorsager, Mette and Hsieh, Yu-Wen and Jug, Florian and Tomancak, Pavel",
editor="Bartoli, Adrien
and Fusiello, Andrea",
title="Registration of Multi-modal Volumetric Images by Establishing Cell Correspondence",
booktitle="Computer Vision -- ECCV 2020 Workshops",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="458--473",
isbn="978-3-030-66415-2"
}
```
