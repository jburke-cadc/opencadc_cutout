import unittest
import logging
import numpy as np

from astropy.io import fits

from .context import opencadc_cutout


class FITSCutoutTestCase(unittest.TestCase):
    def test_pixel_cutout(self):
        logging.getLogger().setLevel('DEBUG')

        # Each point is one pixel out from the box.
        coords = (400.0, 992.0, 1483.0, 2075.0)
        position = (np.average(coords[:2]), np.average(coords[2:]))

        logging.info('\n\nPosition: {}\n\n'.format(position))

        # Size is the length of the sides.  This uses the values from the coordinates to determing the lengths.
        size = (((coords[1] + 1) - (coords[0] + 1)) + 1,
                ((coords[3] + 1) - (coords[2] + 1)) + 1)

        test_subject = opencadc_cutout.cutout_factory.CutoutFactory().factory(
            opencadc_cutout.file_types.FileTypes.FITS)
        cutout_result = test_subject.cutout_from_file(
            'tests/data/test-dao.fits', position, size)

        expected_arr = fits.getdata(
            'tests/data/test-dao-expected-cutout-0__401_993_1484_2076.fits', 0)
        ndarr = cutout_result.data

        self.assertEqual(ndarr.shape, expected_arr.shape,
                         "Shape's don't match.")
        np.testing.assert_array_equal(
            ndarr, expected_arr, "Arrays don't match")

    def test_WCS_cutout(self):
        logging.getLogger().setLevel('DEBUG')


if __name__ == '__main__':
    unittest.main()
