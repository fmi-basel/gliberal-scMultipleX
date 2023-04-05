from unittest import TestCase

from scmultiplex.faim_hcs.hcs.Experiment import Experiment
from scmultiplex.faim_hcs.records.PlateRecord import PlateRecord
from scmultiplex.faim_hcs.records.WellRecord import WellRecord

from src.scmultiplex.utils.exclude_utils import exclude_conditions


class FeatureFunctionTest(TestCase):
    def setUp(self) -> None:
        self.exp = Experiment()
        plates = ["P1", "P2", "P3", "P4"]
        wells = ["A01", "A02", "B03", "B04"]

        self.excluded_plates = ["P2", "P3"]
        self.excluded_wells = ["A01", "B03"]

        # create mini experiment record for this plate and well set
        for p in plates:
            plate_record = PlateRecord(experiment=self.exp, plate_id=p)
            for w in wells:
                # test scenario if one of the plates does not have all the of the wells
                if p != "P4" or w != "B04":
                    WellRecord(plate=plate_record, well_id=w)

    def tearDown(self) -> None:
        pass

    def test_well_number(self):
        # test that the number of wells selected is correct
        results = exclude_conditions(
            self.exp, self.excluded_plates, self.excluded_wells
        )

        self.exp.only_iterate_over_wells(True)
        self.exp.reset_iterator()

        well_count_exp = 0
        for well in self.exp:
            well_count_exp += 1
        print(well_count_exp)

        well_count_res = 0
        for well in results:
            well_count_res += 1
        print(well_count_res)

        # expect 15 wells total in original data
        # expect 3 wells in results

        assert well_count_exp == 15
        assert well_count_res == 3

    def test_select_wells(self):
        # test that the selected wells are in the original data and not in the excluded list
        self.exp.only_iterate_over_wells(True)
        self.exp.reset_iterator()

        # results is list of well records
        results = exclude_conditions(
            self.exp, self.excluded_plates, self.excluded_wells
        )

        for well in results:
            assert well.plate.plate_id not in self.excluded_plates
            assert well.plate.plate_id in ["P1", "P4"]
            assert well.well_id not in self.excluded_wells
            assert well.well_id in ["A02", "B04"]
