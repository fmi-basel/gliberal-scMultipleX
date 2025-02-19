### Purpose
- Calculates a **consensus linking table** across all multiplexing rounds in an HCS OME-Zarr dataset.
- Aligns object labels from all acquisitions to a reference acquisition, ensuring consistent object identities across rounds.
- Stores the resulting consensus table in the reference acquisition directory.

### Outputs
- A **consensus linking table** that maps object labels from all rounds to a single, aligned consensus.
- The table includes:
  - Original object labels from each round (e.g., `R0_label`, `R1_label`, ...).
  - A new consensus label (`consensus_label`) and index (`consensus_index`) for aligned objects.

### Limitations
- Requires pre-existing linking tables generated by a previous linking task (e.g., `Calculate Object Linking`).
- Assumes that the input linking tables follow a consistent structure across rounds.
