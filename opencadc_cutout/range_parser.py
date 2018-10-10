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

import itertools
import logging
import re
import numpy
from .range_parser_error import RangeParserError


class RangeParser(object):

    def __init__(self, delimiter=':'):
        self.logger = logging.getLogger()
        self.logger.setLevel('DEBUG')
        self.delimiter = delimiter

    def parse(self, range_str):
        """
        Parse a string range.
        :param  range_str: The string to parse.

        Example:

        rp = RangeParser()
        rp.parse('1')
        => (1,1)

        rp.parse('99:112')
        => (99,112)
        """
        rs = range_str.strip()
        expected_pattern = re.compile(r'\d+[:\d+]')

        if not re.match(expected_pattern, range_str):
            raise RangeParserError(
                'Invalid range specified.  Should be in the format of {} (i.e. 8:35), or single digit (i.e. 9).'.format(expected_pattern))

        if self.delimiter not in rs:
            rs = rs + self.delimiter + rs  # Turns 7 into 7..7

        start, end = rs.split(self.delimiter)

        if not start or not end:
            raise RangeParserError('Incomplete range specified {}'.format(rs))
        else:
            return (int(start), int(end))

    def expand(self, range_str):
        """
        Expand a string range.
        :param  range_str: The string to expand.

        Example:

        rp = RangeParser()
        rp.parse('1')
        => array([1])

        rp.parse('99:112')
        => array([99,100,101,102,103,104,105,106,107,108,109,110,111,112])
        """
        start, end = self.parse(range_str)

        if not start or not end:
            raise RangeParserError('Incomplete range specified {}'.format(range_str))
        else:
            return [numpy.arange(start=start, stop=end + 1)]
