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

import logging
import numpy as np
import os
import sys
import pytest
import tempfile

from io import BufferedWriter
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, Longitude, Latitude

from .context import opencadc_cutout
from opencadc_cutout.cutout_factory import CutoutFactory

pytest.main(args=['-s', os.path.abspath(__file__)])
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TESTDATA_DIR = os.path.join(THIS_DIR, 'data')
target_file_name = os.path.join(TESTDATA_DIR, 'test-cgps.fits')
expected_cutout_file_name = os.path.join(
    TESTDATA_DIR, 'test-cgps-0__376_397_600_621____.fits')
logger = logging.getLogger()


def test_pixel_cutout():
    logger.setLevel('DEBUG')

    # Each point is one pixel out from the box.
    expected_coords = (375.0, 396.0, 599.0, 620.0)
    position = (np.average(expected_coords[:2]), np.average(
        expected_coords[2:]))
    extension = 0

    logger.info('\n\nPosition: {}\n\n'.format(position))

    # Size is the length of the sides.  This uses the values from the coordinates to determing the lengths.
    size = (((expected_coords[1]) - (expected_coords[0])) + 1,
            ((expected_coords[3]) - (expected_coords[2])) + 1)

    test_subject = CutoutFactory.factory(
        opencadc_cutout.file_types.FileTypes.FITS)

    with fits.open(target_file_name, mode='readonly') as target_hdu_data:
        target_hdu = target_hdu_data[extension]
        wcs = WCS(header=target_hdu.header, naxis=2)
        cutout_result = test_subject.get_cutout(
            target_hdu.data, position, size, wcs)

    with fits.open(expected_cutout_file_name, mode='readonly') as fits_hdu_data:
        primary_hdu = fits_hdu_data[extension]
        expected_wcs = WCS(header=primary_hdu.header, naxis=2)
        expected_arr = np.squeeze(primary_hdu.data)
        ndarr = cutout_result.data
        assert expected_wcs.wcs == cutout_result.wcs.wcs, "WCS doesn't match"

    assert ndarr.shape == expected_arr.shape, "Shapes don't match."
    np.testing.assert_array_equal(
        ndarr, expected_arr, "Arrays don't match")


def test_WCS_cutout():
    logger.setLevel('DEBUG')

    frame = 'ICRS'.lower()
    ra = Longitude(9.0, unit=u.deg)
    dec = Latitude(66.3167, unit=u.deg)
    position = SkyCoord(ra=ra, dec=dec, frame=frame)
    extension = 0

    logger.info('\n\nPosition: {}\n\n'.format(position))

    # Size is the length of the sides.  This uses the values from the coordinates to determing the lengths.
    expected_coords = (375.0, 396.0, 599.0, 620.0)
    size = (((expected_coords[1]) - (expected_coords[0])) + 1,
            ((expected_coords[3]) - (expected_coords[2])) + 1)

    test_subject = CutoutFactory.factory(
        opencadc_cutout.file_types.FileTypes.FITS)

    with fits.open(target_file_name, mode='readonly') as target_hdu_data:
        target_hdu = target_hdu_data[extension]
        wcs = WCS(header=target_hdu.header, naxis=2)
        cutout_result = test_subject.get_cutout(
            target_hdu.data, position, size, wcs)

    with fits.open(expected_cutout_file_name) as fits_hdu_data:
        hdu = fits_hdu_data[extension]
        expected_wcs = WCS(header=hdu.header, naxis=2)
        expected_arr = np.squeeze(hdu.data)
        logger.info('\n\nNAxis in result: {}\n\n'.format(
            cutout_result.wcs.naxis))
        logger.info('\n\nNAxis in expected WCS: {}\n\n'.format(
            expected_wcs.naxis))
        assert expected_wcs.wcs == cutout_result.wcs.wcs, "WCS doesn't match"

    ndarr = cutout_result.data

    assert ndarr.shape == expected_arr.shape, "Shapes don't match."
    np.testing.assert_array_equal(
        ndarr, expected_arr, "Arrays don't match")


def test_cutout_stream():
    logger.setLevel('DEBUG')

    frame = 'ICRS'.lower()
    ra = Longitude(9.0, unit=u.deg)
    dec = Latitude(66.3167, unit=u.deg)
    position = SkyCoord(ra=ra, dec=dec, frame=frame)
    extension = 0

    logger.info('\n\nPosition: {}\n\n'.format(position))

    # Size is the length of the sides.  This uses the values from the coordinates to determing the lengths.
    expected_coords = (375.0, 396.0, 599.0, 620.0)
    size = (((expected_coords[1]) - (expected_coords[0])) + 1,
            ((expected_coords[3]) - (expected_coords[2])) + 1)

    test_subject = CutoutFactory.factory(
        opencadc_cutout.file_types.FileTypes.FITS)

    _, file_name_path = tempfile.mkstemp(suffix='.fits')

    with open(file_name_path, 'wb') as test_file_handle:
        test_subject.cutout(target_file_name, position, size,
                            test_file_handle, extension=extension)
        test_file_handle.close()

    # with fits.open(expected_cutout_file_name) as fits_hdu_data, fits.open(file_name_path) as test_hdu_data:
    #     # assert fits_hdu_data[extension].header == test_hdu_data[extension].header, 'Headers do not match.'
    #     expected_data = fits_hdu_data[extension].data
    #     test_result_data = test_hdu_data[extension].data
    #     assert expected_data == test_result_data, 'Data does not match.'
    #     logger.info('Test header {}'.format(test_hdu_data[extension].header))
    #     assert False
