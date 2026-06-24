from pathlib import Path

import pandas as pd


def load_csv_subset_by_well_timepoint(
    path: str,
    timepoint: str,
    well: str,
    colname_timepoint: str,
    colname_well: str,
    columns_to_keep: list[str] = None,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Read a large CSV in chunks and return only rows matching one
    timepoint and well.

    Optionally subset the output to a specified set of columns.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    timepoint : str
        Timepoint value to keep (e.g. "day1p5").
    well : str
        Well value to keep (e.g. "C01").
    colname_timepoint : str
        Column containing timepoint labels.
    colname_well : str
        Column containing well labels.
    columns_to_keep : list[str], optional
        List of columns to retain in the output DataFrame.
        If None, all columns are retained.

        All requested columns must exist in the CSV, otherwise a
        ValueError is raised.
    chunksize : int
        Number of rows to read per chunk.

    Returns
    -------
    pd.DataFrame
        Subset of the CSV matching the specified timepoint and well.

        If columns_to_keep is provided, only those columns are returned.
        Otherwise, all columns are returned.

        Returns an empty DataFrame if no matching rows are found.

    Raises
    ------
    ValueError
        If any required column (filtering columns or requested output
        columns) is not present in the CSV.
    """

    matched_chunks = []

    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):

        # validate required filter columns
        required_columns = {colname_timepoint, colname_well}

        # validate requested output columns
        if columns_to_keep is not None:
            required_columns.update(columns_to_keep)

        missing = required_columns - set(chunk.columns)

        if missing:
            raise ValueError(
                f"Requested column(s) not found in CSV: {sorted(missing)}\n\n"
                f"Available columns:\n{sorted(chunk.columns.tolist())}"
            )

        # select rows that match both input timepoint and well
        mask_timepoint = chunk[colname_timepoint] == timepoint
        mask_well = chunk[colname_well] == well
        mask = mask_timepoint & mask_well

        subset = chunk.loc[mask]

        # keep only requested columns if provided
        if columns_to_keep is not None:
            subset = subset[columns_to_keep]

        if not subset.empty:
            matched_chunks.append(subset)

    if not matched_chunks:
        return pd.DataFrame()

    return pd.concat(matched_chunks, ignore_index=True)


def parse_timepoint_well_from_zarr_path(zarr_path: str) -> tuple[str, str]:
    """
    Parse a zarr path to extract the timepoint and well.

    Assumes path structure:
        .../<timepoint>/<dataset>.zarr/<row>/<column>/<index>

    Example
    -------
    Input:
        /User/Documents/Fractal/day1p5/220605_151046.zarr/C/02/0

    Output:
        ("day1p5", "C02")
    """

    # Split path into components
    parts = Path(zarr_path).parts

    # Timepoint is always 5th component from the end
    timepoint = parts[-5]

    # Extract well:
    # .../<row>/<column>/<image>
    # e.g. .../C/02/0 -> C02
    well = f"{parts[-3]}{parts[-2]}"

    return timepoint, well
