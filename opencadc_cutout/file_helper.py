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
import os
import re
import regions
import numpy as np
from enum import Enum

from astropy.io import fits
from astropy.nddata.utils import Cutout2D, NoOverlapError
from astropy.wcs import WCS

from regions.core import PixelRegion, SkyRegion, BoundingBox
from regions.shapes.circle import CircleSkyRegion, CirclePixelRegion
from regions.shapes.polygon import PolygonSkyRegion, PolygonPixelRegion

from .no_content_error import NoContentError

UNDESIREABLE_HEADER_KEYS = ['DATASUM', 'CHECKSUM']


class BaseFileHelper(object):
    def __init__(self, file_path):
        self.logger = logging.getLogger()
        self.logger.setLevel('INFO')
        if file_path is None or file_path == '':
            raise ValueError('File path is required.')
        else:
            self.file_path = file_path

    def do_cutout(self, data, position, size, wcs):
        """
        Perform a Cutout of the given data at the given position and size.
        :param data:  The data to cutout from
        :param position:  The position to cutout from
        :param size:  The size in pixels of the cutout
        :param wcs:    The WCS object to use with the cutout to return a copy of the WCS object.

        :return: Cutout2D instance
        """

        # Sanitize the array by removing the single-dimensional entries.
        sanitized_data = np.squeeze(data)
        return Cutout2D(data=sanitized_data, position=position, size=size, wcs=wcs)

    def get_bounding_box_offsets(self, cutout_region):
        shape_type = type(cutout_region).__name__
        if shape_type.startswith('Polygon'):
            return (0, 0, 0, 0)
        else:
            return (-1, 0, -1, 0)

    def get_position_center_offsets(self, cutout_region):
        shape_type = type(cutout_region).__name__
        if shape_type.startswith('Polygon'):
            return (0, -1)
        else:
            return (0, 0)


class FITSHelper(BaseFileHelper):

    def __init__(self, file_path):
        self.logger = logging.getLogger()
        self.logger.setLevel('DEBUG')
        super(FITSHelper, self).__init__(file_path)

    def _post_sanitize_header(self, header):
        """
        Remove headers that don't belong in the cutout output.
        """
        # Remove known keys
        [header.remove(x, ignore_missing=True, remove_all=True)
         for x in UNDESIREABLE_HEADER_KEYS]

        pattern = re.compile(r'PC\d_\d')

        # Remove the CDi_j headers in favour of the PCi_j equivalents
        [header.remove(x[0].replace('PC', 'CD'), ignore_missing=True, remove_all=True)
         for x in list(filter(lambda y: pattern.match(y[0]), header.cards))]

    def _cutout(self, header, data, wcs, position, size, output_writer):
        cutout_result = self.do_cutout(
            data=data, position=position, size=size, wcs=wcs)

        header.update(cutout_result.wcs.to_header())
        self._post_sanitize_header(header)
        fits.append(filename=output_writer, header=header, data=cutout_result.data,
                    overwrite=False, output_verify='silentfix', checksum=False)
        output_writer.flush()

    def _to_bounding_box(self, cutout_region, wcs, extension):
        if isinstance(cutout_region, SkyRegion):
            pixel_cutout_region = cutout_region.to_pixel(wcs)
            try:
                _bounding_box = pixel_cutout_region.bounding_box
            except ValueError as err:
                self.logger.error(
                    'Unable to use HDU at extension {} due to error {}.'.format(extension, err))
                return None
        elif isinstance(cutout_region, PixelRegion):
            _bounding_box = cutout_region.bounding_box
        elif isinstance(cutout_region, BoundingBox):
            _bounding_box = cutout_region
        else:
            raise ValueError(
                'Unsupported region cutout specified: {}'.format(cutout_region))

        box_offsets = self.get_bounding_box_offsets(cutout_region)

        return BoundingBox(
            (_bounding_box.ixmin +
             box_offsets[0]), (_bounding_box.ixmax + box_offsets[1]),
            (_bounding_box.iymin + box_offsets[2]), (_bounding_box.iymax + box_offsets[3]))

    def cutout(self, cutout_region, output_writer):
        with fits.open(self.file_path, mode='readonly', memmap=True) as hdu_list:
            cutouts_found = 0
            hdu_count = len(hdu_list)
            hdu_iter = iter(hdu_list[:])
            if hdu_count > 1:
                self.logger.debug('MEF has {} HDUs'.format(hdu_count))
                primary_hdu = next(hdu_iter)
                fits.append(filename=output_writer, header=primary_hdu.header, data=None, overwrite=False,
                            output_verify='silentfix', checksum=False)
                start_index = 1
            else:
                start_index = 0

            for extension, hdu in enumerate(hdu_iter, start=start_index):
                if np.any(hdu.data) and hdu.data.size > 0:
                    header = hdu.header
                    wcs = WCS(header=header, naxis=2)
                    bounding_box = self._to_bounding_box(
                        cutout_region, wcs, extension)

                    if bounding_box:
                        position_offsets = self.get_position_center_offsets(
                            cutout_region)
                        box_center = bounding_box.to_region().center
                        position = (
                            box_center.x + position_offsets[0], box_center.y + position_offsets[1])
                        size = bounding_box.shape
                        self.logger.debug(
                            'Position: {}\nSize {}'.format(position, size))
                        try:
                            self._cutout(header=header, data=hdu.data, wcs=wcs,
                                         position=position, size=size, output_writer=output_writer)
                            self.logger.debug(
                                'Cutting out from extension {}'.format(extension))
                            cutouts_found += 1
                        except NoOverlapError:
                            self.logger.info(
                                'No overlap found for extension {}'.format(extension))

            if cutouts_found == 0:
                raise NoContentError('No content (arrays do not overlap).')


class FileTypeHelpers(Enum):
    FITS = FITSHelper


class FileHelperFactory(object):

    def get_instance(self, file_path):
        _, extension = os.path.splitext(file_path)
        helper_class = FileTypeHelpers[extension.split('.')[1].upper()].value
        return helper_class(file_path)
