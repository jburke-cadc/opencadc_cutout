# -*- coding: utf-8 -*-
import logging
import numpy as np

from astropy.nddata import Cutout2D


class Cutout(object):

    def __init__(self):
        logging.getLogger().setLevel('INFO')
        self.logger = logging.getLogger(__name__)

    def cutout_from_data(self, data, position, size, wcs=None):
        """
        Perform a Cutout of the given data at the given position and size.
        :param data:  The data to cutout from
        :param position:  The position to cutout from
        :param size:  The size in pixels of the cutout
        :param wcs:    The WCS object to use with the cutout to return a copy of the WCS object.

        :return: Cutout stream
        """
        sanitized_data = np.squeeze(data)
        return Cutout2D(data=sanitized_data, position=position, size=size, wcs=wcs)
