### Purpose
- Expands segmented **labels** in 2D or 3D images **without overlap**.
- Supports expansion by a fixed pixel distance or dynamically based on **label size**.
- Optionally masks expanded labels using **parent objects** to prevent spillover.
- Outputs an expanded label image and preserves non-overlapping object boundaries.

### Outputs
- A new **expanded label image** saved with an `_expanded` suffix.

### Limitations
- If masking by parent is enabled, the parent object label image must be provided.
- Expansion beyond object boundaries may be clipped, depending on the surrounding labels and image dimensions.
