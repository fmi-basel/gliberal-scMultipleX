from fractal_tasks_core.channels import ChannelInputModel
from pydantic import BaseModel


class ZIlluminationChannelInputModel(ChannelInputModel):
    """
    Extended version of ChannelInputModel with an additional attribute.

    Attributes:
        z_illum_table: Stores illumination correction data for different z-levels.
    """

    z_correction_table: str = None
    background_intensity: int = 0


class InitArgsIllumination(BaseModel):
    """
    Registration init args.

    Passed from `image_based_registration_hcs_init` to
    `calculate_registration_image_based`.

    Attributes:
        correction_zarr_url: zarr_url for the correction round
    """

    correction_zarr_url: str
