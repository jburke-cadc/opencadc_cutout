# -*- coding: utf-8 -*-
# ***********************************************************************
# ******************  CANADIAN ASTRONOMY DATA CENTRE  *******************
# *************  CENTRE CANADIEN DE DONNÉES ASTRONOMIQUES  **************
#
#  (c) 2018.                            (c) 2018.
#  Government of Canada                 Gouvernement du Canada
#  National Research Council            Conseil national de recherches
#  Ottawa, Canada, K1A 0R6              Ottawa, Canada, K1A 0R6
#  All rights reserved                  Tous droits réservés
#
#  NRC disclaims any warranties,        Le CNRC dénie toute garantie
#  expressed, implied, or               énoncée, implicite ou légale,
#  statutory, of any kind with          de quelque nature que ce
#  respect to the software,             soit, concernant le logiciel,
#  including without limitation         y compris sans restriction
#  any warranty of merchantability      toute garantie de valeur
#  or fitness for a particular          marchande ou de pertinence
#  purpose. NRC shall not be            pour un usage particulier.
#  liable in any event for any          Le CNRC ne pourra en aucun cas
#  damages, whether direct or           être tenu responsable de tout
#  indirect, special or general,        dommage, direct ou indirect,
#  consequential or incidental,         particulier ou général,
#  arising from the use of the          accessoire ou fortuit, résultant
#  software.  Neither the name          de l'utilisation du logiciel. Ni
#  of the National Research             le nom du Conseil National de
#  Council of Canada nor the            Recherches du Canada ni les noms
#  names of its contributors may        de ses  participants ne peuvent
#  be used to endorse or promote        être utilisés pour approuver ou
#  products derived from this           promouvoir les produits dérivés
#  software without specific prior      de ce logiciel sans autorisation
#  written permission.                  préalable et particulière
#                                       par écrit.
#
#  This file is part of the             Ce fichier fait partie du projet
#  OpenCADC project.                    OpenCADC.
#
#  OpenCADC is free software:           OpenCADC est un logiciel libre ;
#  you can redistribute it and/or       vous pouvez le redistribuer ou le
#  modify it under the terms of         modifier suivant les termes de
#  the GNU Affero General Public        la “GNU Affero General Public
#  License as published by the          License” telle que publiée
#  Free Software Foundation,            par la Free Software Foundation
#  either version 3 of the              : soit la version 3 de cette
#  License, or (at your option)         licence, soit (à votre gré)
#  any later version.                   toute version ultérieure.
#
#  OpenCADC is distributed in the       OpenCADC est distribué
#  hope that it will be useful,         dans l’espoir qu’il vous
#  but WITHOUT ANY WARRANTY;            sera utile, mais SANS AUCUNE
#  without even the implied             GARANTIE : sans même la garantie
#  warranty of MERCHANTABILITY          implicite de COMMERCIALISABILITÉ
#  or FITNESS FOR A PARTICULAR          ni d’ADÉQUATION À UN OBJECTIF
#  PURPOSE.  See the GNU Affero         PARTICULIER. Consultez la Licence
#  General Public License for           Générale Publique GNU AfferoF
#  more details.                        pour plus de détails.
#
#  You should have received             Vous devriez avoir reçu une
#  a copy of the GNU Affero             copie de la Licence Générale
#  General Public License along         Publique GNU Affero avec
#  with OpenCADC.  If not, see          OpenCADC ; si ce n’est
#  <http://www.gnu.org/licenses/>.      pas le cas, consultez :
#                                       <http://www.gnu.org/licenses/>.
#
#  $Revision: 1 $
#
# ***********************************************************************
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import numpy as np
import os
import sys
import pytest
import tempfile
import regions

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from regions.core import PixCoord, BoundingBox
from regions.shapes.circle import CirclePixelRegion, CircleSkyRegion
from regions.shapes.polygon import PolygonPixelRegion, PolygonSkyRegion
from astropy.coordinates import SkyCoord, Longitude, Latitude

from .context import opencadc_cutout
from opencadc_cutout.core import Cutout
from opencadc_cutout.no_content_error import NoContentError


pytest.main(args=['-s', os.path.abspath(__file__)])
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TESTDATA_DIR = os.path.join(THIS_DIR, 'data')
target_file_name = os.path.join(TESTDATA_DIR, 'test-cgps.fits')
expected_circle_cutout_file_name = os.path.join(
    TESTDATA_DIR, 'test-cgps-0__376_397_600_621____.fits')
expected_polygon_cutout_file_name = os.path.join(
    TESTDATA_DIR, 'test-cgps-polygon-cutout.fits')
logger = logging.getLogger()


def _check_circle_output_file(cutout_regions, extension=0):
    test_subject = Cutout()

    # To debug the resulting FITS files when running with Docker, set dir='/usr/src/app'
    _, cutout_file_name_path = tempfile.mkstemp(suffix='.fits')

    # Write out a test file with the test result FITS data.
    with open(cutout_file_name_path, 'wb') as test_file_handle:
        test_subject.cutout(target_file_name, cutout_regions, test_file_handle)
        test_file_handle.close()

    with fits.open(expected_circle_cutout_file_name, mode='readonly') as expected_hdu_data, fits.open(cutout_file_name_path, mode='readonly') as result_hdu_data:
        expected_hdu = expected_hdu_data[extension]
        result_hdu = result_hdu_data[extension]
        expected_wcs = WCS(header=expected_hdu.header, naxis=2)
        result_wcs = WCS(header=result_hdu.header, naxis=2)

        fits_diff = fits.FITSDiff(expected_hdu_data, result_hdu_data)
        np.testing.assert_array_equal(
            (), fits_diff.diff_hdu_count, 'HDU count diff should be empty.')
        np.testing.assert_array_equal(
            expected_wcs.wcs.crpix, result_wcs.wcs.crpix, 'Wrong CRPIX values.')
        np.testing.assert_array_equal(
            np.squeeze(expected_hdu.data), result_hdu.data, 'Arrays do not match.')


def test_bounding_box_pixel_cutout():
    """
    Test a simple bounding box matching the parameters of the test file.
    """
    logger.setLevel('DEBUG')
    extension = 0
    cutout_region = BoundingBox(376, 397, 600, 621)
    cutout_regions = [cutout_region]
    _check_circle_output_file(cutout_regions, extension=extension)


def test_circle_pixel_cutout():
    """
    Test a Pixel (PixCoord) circle.
    """
    logger.setLevel('DEBUG')
    extension = 0
    cutout_region = CirclePixelRegion(
        PixCoord(x=386.2794725953952, y=610.3341259408464), radius=10.0000003)
    cutout_regions = [cutout_region]
    _check_circle_output_file(cutout_regions, extension=extension)


def test_circle_wcs_cutout():
    """
    Test a WCS (SkyCoord) circle.
    """
    logger.setLevel('DEBUG')
    extension = 0
    frame = 'ICRS'.lower()
    ra = Longitude(9.0, unit=u.deg)
    dec = Latitude(66.3167, unit=u.deg)
    radius = u.Quantity(0.05, unit=u.deg)
    sky_position = SkyCoord(ra=ra, dec=dec, frame=frame)
    cutout_region = CircleSkyRegion(sky_position, radius=radius)
    cutout_regions = [cutout_region]
    _check_circle_output_file(cutout_regions, extension=extension)


def _check_polygon_output_file(cutout_regions, extension=0):
    test_subject = Cutout()

    # To debug the resulting FITS files when running with Docker, set dir='/usr/src/app'
    _, cutout_file_name_path = tempfile.mkstemp(
        prefix='test_polygon_', suffix='.fits', dir='/usr/src/app')

    logger.info('Testing with {}'.format(cutout_file_name_path))

    # Write out a test file with the test result FITS data.
    with open(cutout_file_name_path, 'wb') as test_file_handle:
        test_subject.cutout(target_file_name, cutout_regions, test_file_handle)
        test_file_handle.close()

    with fits.open(expected_polygon_cutout_file_name, mode='readonly') as expected_hdu_data, fits.open(cutout_file_name_path, mode='readonly') as result_hdu_data:
        expected_hdu = expected_hdu_data[extension]
        result_hdu = result_hdu_data[extension]
        expected_wcs = WCS(header=expected_hdu.header, naxis=2)
        result_wcs = WCS(header=result_hdu.header, naxis=2)

        fits_diff = fits.FITSDiff(expected_hdu_data, result_hdu_data)
        np.testing.assert_array_equal(
            (), fits_diff.diff_hdu_count, 'HDU count diff should be empty.')
        # np.testing.assert_array_equal(
            # expected_wcs.wcs.crpix, result_wcs.wcs.crpix, 'Wrong CRPIX values.')
        np.testing.assert_array_equal(
            np.squeeze(expected_hdu.data), result_hdu.data, 'Arrays do not match.')


def test_polygon_pixel_cutout():
    """
    Test a Pixel (PixCoord) polygon.
    """
    logger.setLevel('DEBUG')
    extension = 0
    x, y = [376, 600, 405, 216], [397, 621, 740, 668]
    vertices = PixCoord(x=x, y=y)
    cutout_region = PolygonPixelRegion(vertices=vertices)
    cutout_regions = [cutout_region]
    _check_polygon_output_file(cutout_regions, extension=extension)


def test_polygon_wcs_cutout():
    """
    Test a WCS (SkyCoord) polygon.
    """
    logger.setLevel('DEBUG')
    extension = 0
    frame = 'ICRS'.lower()
    ra, dec = [9.27403199, 6.3392766, 8.66286563, 11.10050874] * \
        u.deg, [65.25472033, 66.28375562, 66.95796478, 66.64119763] * u.deg
    vertices = SkyCoord(ra=ra, dec=dec, frame=frame)
    cutout_region = PolygonSkyRegion(vertices)
    cutout_regions = [cutout_region]
    _check_polygon_output_file(cutout_regions, extension=extension)


def test_no_content_cutout():
    """
    Test an invalid cutout.
    """
    frame = 'ICRS'.lower()
    logger.setLevel('DEBUG')
    extension = 0
    cutout_region = PolygonSkyRegion(
        SkyCoord([3, 4, 3] * u.deg, [3, 4, 4] * u.deg, frame=frame))
    cutout_regions = [cutout_region]
    try:
        _check_polygon_output_file(cutout_regions, extension=extension)
        assert False, 'Should raise NoContentError.'
    except NoContentError as err:
        expected_message = 'No content (arrays do not overlap).'
        result_message = '{}'.format(err)
        assert expected_message == result_message, 'Wrong error (expected {}).'.format(
            expected_message)
