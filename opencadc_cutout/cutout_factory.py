# Factory/shapefact1/ShapeFactory1.py
# A simple static factory method.
from __future__ import generators
import random
from .fits import FITSCutout
from .file_types import FileTypes


class CutoutFactory(object):

    # Create based on supported FileTypes
    def factory(file_type):
        """
        Generate a new Cutout instance.
        :param file_type:  An instance of the FileType Enum.
        """
        if file_type == FileTypes.FITS:
            return FITSCutout()
        assert 0, "Bad cutout creation: " + file_type
    factory = staticmethod(factory)
