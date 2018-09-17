import logging
import numpy as np
import os
import sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, Longitude, Latitude

from .context import opencadc_cutout
from opencadc_cutout.cutout_factory import CutoutFactory

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TESTDATA_DIR = os.path.join(THIS_DIR, 'data')
target_file_name = os.path.join(TESTDATA_DIR, 'test-cgps.fits')
expected_cutout_file_name = os.path.join(
    TESTDATA_DIR, 'test-cgps-0__376_397_600_621____.fits')


def test_pixel_cutout():
    logging.getLogger().setLevel('DEBUG')

    # Each point is one pixel out from the box.
    coords = (375.0, 396.0, 599.0, 620.0)
    position = (np.average(coords[:2]), np.average(coords[2:]))
    extension = 0

    logging.info('\n\nPosition: {}\n\n'.format(position))

    # Size is the length of the sides.  This uses the values from the coordinates to determing the lengths.
    size = (((coords[1]) - (coords[0])) + 1,
            ((coords[3]) - (coords[2])) + 1)

    test_subject = CutoutFactory.factory(
        opencadc_cutout.file_types.FileTypes.FITS)

    cutout_result = test_subject.cutout(target_file_name, position, size)

    with fits.open(expected_cutout_file_name) as fits_hdu_data:
        primary_hdu = fits_hdu_data[extension]
        expected_arr = np.squeeze(primary_hdu.data)
        ndarr = cutout_result.data

    assert cutout_result.wcs == None, 'Should omit WCS.'
    assert ndarr.shape == expected_arr.shape, "Shapes don't match."
    np.testing.assert_array_equal(
        ndarr, expected_arr, "Arrays don't match")


def test_WCS_cutout():
    logging.getLogger().setLevel('DEBUG')

    frame = 'ICRS'.lower()
    ra = Longitude(9.0, unit=u.deg)
    dec = Latitude(66.3167, unit=u.deg)
    position = SkyCoord(ra=ra, dec=dec, frame=frame)
    extension = 0

    logging.info('\n\nPosition: {}\n\n'.format(position))

    # Size is the length of the sides.  This uses the values from the coordinates to determing the lengths.
    size = (22, 22)

    test_subject = CutoutFactory.factory(
        opencadc_cutout.file_types.FileTypes.FITS)
    cutout_result = test_subject.cutout(target_file_name, position, size)

    with fits.open(expected_cutout_file_name) as fits_hdu_data:
        hdu = fits_hdu_data[extension]
        expected_wcs = WCS(header=hdu.header, naxis=2)
        expected_arr = np.squeeze(hdu.data)
        logging.info('\n\nNAxis in result: {}\n\n'.format(
            cutout_result.wcs.naxis))
        logging.info('\n\nNAxis in expected WCS: {}\n\n'.format(
            expected_wcs.naxis))
        assert expected_wcs.wcs == cutout_result.wcs.wcs, "WCS doesn't match"

    ndarr = cutout_result.data

    assert ndarr.shape == expected_arr.shape, "Shapes don't match."
    np.testing.assert_array_equal(
        ndarr, expected_arr, "Arrays don't match")
