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
import regions
import numpy as np
from enum import Enum

from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS

from regions.core import PixelRegion, SkyRegion, BoundingBox


class BaseFileHelper(object):
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


class FITSHelper(BaseFileHelper):

    def __init__(self, file_path):
        self.logger = logging.getLogger()
        self.logger.setLevel('DEBUG')
        if file_path is None or file_path == '':
            raise ValueError('FITS file path is required.')
        else:
            self.file_path = file_path
        super(FITSHelper, self).__init__()

    def _get_extensions(self, kwargs):
        if 'extensions' in kwargs:
            return kwargs['extensions']
        elif 'extension' in kwargs:
            return [kwargs['extension']]
        else:
            return [0]

    def cutout(self, cutout_region, output_writer, **kwargs):
        with fits.open(self.file_path, mode='readonly') as fits_data:
            extension = self._get_extensions(kwargs)[0]
            hdu = fits_data[extension]
            header = hdu.header
            wcs = WCS(header=header, naxis=2)

            if isinstance(cutout_region, SkyRegion):
                _bounding_box = cutout_region.to_pixel(wcs).bounding_box
            elif isinstance(cutout_region, PixelRegion):
                _bounding_box = cutout_region.bounding_box
            elif isinstance(cutout_region, BoundingBox):
                _bounding_box = cutout_region
            else:
                raise ValueError(
                    'Unsupported region cutout specified: {}'.format(cutout_region))

            bounding_box = BoundingBox.from_float(
                xmin=(_bounding_box.ixmin - 1.0), xmax=(_bounding_box.ixmax - 1.0),
                ymin=(_bounding_box.iymin - 1.0), ymax=(_bounding_box.iymax - 1.0))
            box_center = bounding_box.to_region().center
            position = (box_center.x, box_center.y)
            size = bounding_box.shape

            cutout_result = self.do_cutout(
                data=hdu.data, position=position, size=size, wcs=wcs)

            header.update(cutout_result.wcs.to_header())
            fits.writeto(filename=output_writer, header=header, data=cutout_result.data,
                         overwrite=True, output_verify='silentfix', checksum=False)


class FileTypeHelpers(Enum):
    FITS = FITSHelper


class FileHelperFactory(object):

    def get_instance(self, file_path):
        _, extension = os.path.splitext(file_path)
        helper_class = FileTypeHelpers[extension.split('.')[1].upper()].value
        return helper_class(file_path)
