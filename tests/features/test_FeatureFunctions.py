from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from scmultiplex.features.FeatureFunctions import fixed_percentiles


class FeatureFunctionTest(TestCase):
    def test_fixed_percentiles(self):
        mask = np.array(
            [[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]], dtype=bool
        )
        img = np.array(
            [[0, 90, 4, 5], [3, 56, 2, 76], [4, 6, 9, 111], [54, 245, 34, 360]]
        )
        results = fixed_percentiles(mask, img)
        expected_result = np.array([5.0, 9.0, 76.0, 94.2, 102.6, 109.32])

        assert_array_equal(results, expected_result)
