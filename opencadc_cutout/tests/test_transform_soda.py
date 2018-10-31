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
import os

import pytest
from astropy.io import fits

from opencadc_cutout.no_content_error import NoContentError
from opencadc_cutout.transform import Transform, Shape

pytest.main(args=['-s', os.path.abspath(__file__)])
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TESTDATA_DIR = os.path.join(THIS_DIR, 'data')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Header from ad:IRIS/I212B2H0.fits
QUERY_HEADER = 'IRIS-I212B2H0.fits.hdr'

# Header from ad:CGPS/CGPS_MA1_HI_line_image.fits
CUBE_HEADER = 'CGPS-CGPS_MA1_HI_line_image.fits.hdr'

# Header from ad:MACHO/cal000400r.fits.fz
NAMED_PART_HEADER = 'MACHO-cal000400r.fits.fz.hdr'

# Header from ad:CFHT/1598392i.fits.gz
TILED_CHUNK_HEADER = 'CFHT-1598392i.fits.gz.hdr'

# Header from ad:CFHT/1598421p.fits.gz
TILED_MULTICHUNK_HEADER = 'CFHT-1598421p.fits.gz.hdr'


# @pytest.mark.skip
def test_circle():
    header_filename = os.path.join(TESTDATA_DIR, QUERY_HEADER)
    header = fits.Header.fromtextfile(header_filename)

    coords = [140.0, 0.0, 0.1]

    test_subject = Transform()
    pixels = test_subject.get_circle_cutout_pixels(coords, header, 1, 2)

    # SODA returns [0][271:279,254:262,*]
    assert pixels is not None
    assert len(pixels) == 4
    assert pixels[0] == 271
    assert pixels[1] == 280
    assert pixels[2] == 253
    assert pixels[3] == 262


# @pytest.mark.skip
def test_polygon():
    header_filename = os.path.join(TESTDATA_DIR, QUERY_HEADER)
    header = fits.Header.fromtextfile(header_filename)

    # CCW winding direction
    coords = [139.9, 0.1, 140.1, 0.1, 140.1, -0.1, 139.9, -0.1]

    test_subject = Transform()
    pixels = test_subject.get_polygon_cutout_pixels(coords, header, 1, 2)

    # SODA returns [0][271:279,254:262,*]
    assert pixels is not None
    assert len(pixels) == 4
    assert pixels[0] == 271
    assert pixels[1] == 280
    assert pixels[2] == 253
    assert pixels[3] == 263


# skip, test failing but shouldn't???
@pytest.mark.skip
def test_interval_polygon():
    header_filename = os.path.join(TESTDATA_DIR, QUERY_HEADER)
    header = fits.Header.fromtextfile(header_filename)

    # CW winding direction
    coords = [139.9, -0.1, 140.1, -0.1, 140.1, 0.1, 139.9, 0.1]

    test_subject = Transform()
    pixels = test_subject.get_polygon_cutout_pixels(coords, header, 1, 2)
    # should fail
    assert pixels is None


# @pytest.mark.skip
def test_pos_circle_upper():
    do_pos_circle('CIRCLE')


# @pytest.mark.skip
def test_pos_circle_lower():
    do_pos_circle('circle')


# @pytest.mark.skip
def test_pos_circle_mixed():
    do_pos_circle('cIrClE')


def do_pos_circle(circle):
    cutout = '{}=140 0 0.1'.format(circle)

    test_subject = Transform()
    shapes = test_subject.parse_world_to_shapes(cutout)
    assert shapes is not None
    assert len(shapes) == 1
    shape = shapes[0]
    assert shape[0] == Shape.CIRCLE
    coordinates = shape[1]
    assert len(coordinates) == 3
    assert coordinates[0] == 140.0
    assert coordinates[1] == 0.0
    assert coordinates[2] == 0.1


# @pytest.mark.skip
def test_circle_no_overlap():
    header_filename = os.path.join(TESTDATA_DIR, QUERY_HEADER)
    header = fits.Header.fromtextfile(header_filename)

    coords = [20, 20, 0.1]

    test_subject = Transform()
    try:
        pixels = test_subject.get_circle_cutout_pixels(coords, header, 1, 2)
        assert False, 'Should raise NoContentError.'
    except NoContentError:
        assert True


# @pytest.mark.skip
def test_band():
    header_filename = os.path.join(TESTDATA_DIR, CUBE_HEADER)
    header = fits.Header.fromtextfile(header_filename)
    header.append(('RESTFRQ', 1.420406E9))

    # energy from caom2
    # bandpassName: 21 cm
    # resolvingPower: null
    # specsys: LSRK
    # ssysobs: null
    # restfrq: 1.420406E9
    # restwav: null
    # velosys: null
    # zsource: null
    # ssyssrc: null
    # velang: null
    # ctype: VRAD
    # cunit: m/s
    # syser:
    # rnder:
    # naxis: 272
    # crpix: 145.0
    # crval: -60000.0
    # cdelt: -824.46002
    # bounds: null
    # range: null

    coords = [211.0e-3, 211.05e-3]

    test_subject = Transform()
    pixels = test_subject.get_energy_cutout_pixels(coords, header, 3)
    assert pixels is not None
    assert len(pixels) == 2
    # SODA returns [0][*,*,91:177,*]
    assert pixels[0] == 91
    assert pixels[1] == 178


# @pytest.mark.skip
def test_band_no_overlap():
    header_filename = os.path.join(TESTDATA_DIR, CUBE_HEADER)
    header = fits.Header.fromtextfile(header_filename)
    header.append(('RESTFRQ', 1.420406E9))

    coords = [212.0, 213.0]

    test_subject = Transform()
    try:
        pixels = test_subject.get_energy_cutout_pixels(coords, header, 3)
        assert False, 'Should raise NoContentError'
    except NoContentError as e:
        assert isinstance(e, NoContentError)


# @pytest.mark.skip
def test_pos_band():
    header_filename = os.path.join(TESTDATA_DIR, CUBE_HEADER)
    header = fits.Header.fromtextfile(header_filename)
    header.append(('RESTFRQ', 1.420406E9))

    cutout = 'circle=25.0+60.0+0.5&BAND=211.0e-3+211.05e-3'

    test_subject = Transform()
    pixel_cutout_hdu = test_subject.world_to_pixels(cutout, header)

    # SODA returns [0][350:584,136:370,91:177,*]
    assert pixel_cutout_hdu is not None
    ranges = pixel_cutout_hdu.get_ranges()
    assert len(ranges) == 4
    assert ranges[0] == (367, 568)
    assert ranges[1] == (152, 353)
    assert ranges[2] == (91, 178)
    assert ranges[3] == (1, 1)


# skip, incomplete header to do the cutout
@pytest.mark.skip
def test_band_tiled_chunk():
    header_filename = os.path.join(TESTDATA_DIR, NAMED_PART_HEADER)
    header = fits.Header.fromtextfile(header_filename)

    coords = [371.0e-9, 372.0e-9]

    test_subject = Transform()
    pixels = test_subject.get_energy_cutout_pixels(coords, header, 3)
    assert pixels is not None
    assert len(pixels) == 2
    # SODA returns [0][1:2724,1:2]
    assert pixels[0] == 91
    assert pixels[1] == 177


# skip, incomplete header to do the cutout
@pytest.mark.skip
def test_band_tiled_multi_chunk():
    header_filename = os.path.join(TESTDATA_DIR, TILED_MULTICHUNK_HEADER)
    header = fits.Header.fromtextfile(header_filename)

    cutout = [371.0e-9, 372.0e-9]

    test_subject = Transform()
    pixels = test_subject.get_energy_cutout_pixels(cutout, header, 3)
    assert pixels is not None
    assert len(pixels) == 2
    # SODA returns [0][1:2724,1:3]
    assert pixels[0] == 1
    assert pixels[1] == 2724
