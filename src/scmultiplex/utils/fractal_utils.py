from fractal_tasks_core.channels import ChannelInputModel
from pydantic import BaseModel


class FeatureChannelInputModel(ChannelInputModel):
    """
    Extended version of ChannelInputModel with an additional attribute.

    Attributes:
        threshold_intensity: Intensity threshold above which to classify pixels as positive.
    """

    threshold_intensity: int = 0


class ZIlluminationChannelInputModel(ChannelInputModel):
    """
    Extended version of ChannelInputModel with an additional attribute.

    Attributes:
        z_correction_table: Name of table that stores illumination correction data for different z-levels.
        background_intensity: Background intensity to subtract prior to applying z-correction.
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
