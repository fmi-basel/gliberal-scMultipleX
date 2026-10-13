### Purpose
- Computes **spherical harmonics** for 3D segmented objects in a label image using the **aics_shparam** library.
- Calculates and analyzes the **shape features** of objects, including reconstruction error.
- Outputs spherical harmonic coefficients and optionally saves reconstructed surface meshes.

### Outputs
- A **feature table** containing spherical harmonic coefficients and reconstruction error (**MSE**) per object.
- Optionally, the **computed surface mesh** and the **reconstructed mesh** (from harmonics), saved as `.stl` files in a new `meshes` folder.

### Limitations
- Input label image must contain 3D segmented objects; neighboring objects are removed by masking.
- The accuracy of spherical harmonics depends on the chosen **maximum degree (`lmax`)** and input label quality.
- Mesh reconstruction might smooth out fine details in highly complex shapes.
