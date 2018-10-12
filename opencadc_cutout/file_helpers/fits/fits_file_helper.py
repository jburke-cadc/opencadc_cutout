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
import re
import numpy as np
import time

from copy import copy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import NoOverlapError
from ..base_file_helper import BaseFileHelper
from ...no_content_error import NoContentError


def current_milli_time(): return int(round(time.time() * 1000))


# Remove the DQ1 and DQ2 headers until the issue with wcslib is resolved:
# https://github.com/astropy/astropy/issues/7828
UNDESIREABLE_HEADER_KEYS = ['DQ1', 'DQ2']


class FITSHelper(BaseFileHelper):

    def __init__(self, file_path):
        self.logger = logging.getLogger()
        self.logger.setLevel('DEBUG')
        super(FITSHelper, self).__init__(file_path)

    def _post_sanitize_header(self, header, original_header):
        """
        Remove headers that don't belong in the cutout output.
        """
        # Remove known keys
        [header.remove(x, ignore_missing=True, remove_all=True)
         for x in UNDESIREABLE_HEADER_KEYS]

        # If the PCi_j matrices were added, then ensure the CDi_j matrices are removed.
        pattern = re.compile(r'PC\d_\d')

        # Remove the CDi_j headers in favour of the PCi_j equivalents
        [header.remove(x[0].replace('PC', 'CD'), ignore_missing=True, remove_all=True)
         for x in list(filter(lambda y: pattern.match(y[0]), header.cards))]

        # If a WCSAXES card exists, ensure that it comes before the CTYPE1 card.
        wcsaxes_keyword = 'WCSAXES'
        ctype1_keyword = 'CTYPE1'

        # Only proceed with this if both the WCSAXES and CTYPE1 cards are present.
        if header.get(wcsaxes_keyword) and header.get(ctype1_keyword):
            wcsaxes_index = header.index(wcsaxes_keyword)
            ctype1_index = header.index('CTYPE1')

            if wcsaxes_index > ctype1_index:
                existing_wcsaxes_value = header.get(wcsaxes_keyword)
                header.remove(wcsaxes_keyword)
                header.insert(
                    ctype1_index, (wcsaxes_keyword, existing_wcsaxes_value))

    def _cutout(self, header, data, cutout_dimension, wcs, output_writer):
        cutout_result = self.do_cutout(
            data=data, cutout_dimension=cutout_dimension, wcs=wcs)

        original_header = copy(header)
        header.update(cutout_result.wcs.to_header())
        self._post_sanitize_header(header, original_header)
        fits.append(filename=output_writer, header=header, data=cutout_result.data,
                    overwrite=False, output_verify='exception', checksum='remove')
        output_writer.flush()

    def cutout(self, cutout_dimension, output_writer):
        extension = cutout_dimension.extension
        start_time = current_milli_time()

        self.logger.debug('Starting load at {}'.format(start_time))
        hdu = fits.getdata(self.file_path, header=True,
                           ext=extension, memmap=True, do_not_scale_image_data=True)
        end_time = current_milli_time()
        self.logger.debug(
            'End load at {} - ({} seconds)'.format(end_time, (end_time - start_time) / 1000))

        header = hdu[1]
        wcs = WCS(header=header)
        try:
            self._cutout(header=header, data=hdu[0],
                         cutout_dimension=cutout_dimension, wcs=wcs, output_writer=output_writer)
            self.logger.debug(
                'Cutting out from extension {}'.format(extension))
        except NoOverlapError:
            self.logger.error(
                'No overlap found for extension {}'.format(extension))
            raise NoContentError('No content (arrays do not overlap).')
