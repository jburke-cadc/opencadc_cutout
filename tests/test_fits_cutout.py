import unittest
import logging
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, Longitude, Latitude

from .context import opencadc_cutout
from opencadc_cutout.cutout_factory import CutoutFactory


class FITSCutoutTestCase(unittest.TestCase):
    def test_pixel_cutout(self):
        logging.getLogger().setLevel('DEBUG')

        target_file_name = 'tests/data/test-dao.fits'
        expected_cutout_file_name = 'tests/data/test-dao-expected-cutout-0__401_993_1484_2076.fits'

        # Each point is one pixel out from the box.
        coords = (400.0, 992.0, 1483.0, 2075.0)
        position = (np.average(coords[:2]), np.average(coords[2:]))

        logging.info('\n\nPosition: {}\n\n'.format(position))

        # Size is the length of the sides.  This uses the values from the coordinates to determing the lengths.
        size = (((coords[1] + 1) - (coords[0] + 1)) + 1,
                ((coords[3] + 1) - (coords[2] + 1)) + 1)

        test_subject = CutoutFactory.factory(
            opencadc_cutout.file_types.FileTypes.FITS)

        cutout_result = test_subject.cutout(target_file_name, position, size)

        with fits.open(expected_cutout_file_name) as fits_hdu_data:
            primary_hdu = fits_hdu_data[0]
            expected_arr = primary_hdu.data           
            ndarr = cutout_result.data

        self.assertIsNone(cutout_result.wcs, "Should omit WCS.")
        self.assertEqual(ndarr.shape, expected_arr.shape,
                         "Shapes don't match.")
        np.testing.assert_array_equal(
            ndarr, expected_arr, "Arrays don't match")

    def test_WCS_cutout(self):
        logging.getLogger().setLevel('DEBUG')

        frame = 'ICRS'.lower()
        target_file_name = 'tests/data/test-dao.fits'
        expected_cutout_file_name = 'tests/data/test-dao-expected-cutout-0__401_993_1484_2076.fits'
        ra = Longitude(210.80227, unit=u.deg)
        dec = Latitude(54.34895, unit=u.deg)
        position = SkyCoord(ra=ra, dec=dec, frame=frame)

        logging.info('\n\nPosition: {}\n\n'.format(position))

        # Size is the length of the sides.  This uses the values from the coordinates to determing the lengths.
        size = ((993 - 401) + 1, (2076 - 1484) + 1)
        test_subject = CutoutFactory.factory(opencadc_cutout.file_types.FileTypes.FITS)
        cutout_result = test_subject.cutout(target_file_name, position, size)

        with fits.open(expected_cutout_file_name) as fits_hdu_data:
            primary_hdu = fits_hdu_data[0]
            expected_arr = primary_hdu.data           
            expected_wcs = WCS(primary_hdu.header)
            ndarr = cutout_result.data            

        expected_arr = fits.getdata(expected_cutout_file_name, 0)
        ndarr = cutout_result.data

        self.assertEqual(ndarr.shape, expected_arr.shape, "Shapes don't match.")
        self.assertEqual(cutout_result.wcs, expected_wcs, "WCS doesn't match")
        np.testing.assert_array_equal(
            ndarr, expected_arr, "Arrays don't match")


if __name__ == '__main__':
    unittest.main()
