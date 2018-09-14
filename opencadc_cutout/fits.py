from .core import Cutout

from astropy.io import fits


class FITSCutout(Cutout):

    def __init__(self):
        super(FITSCutout, self).__init__()

    def cutout_from_file(self, file_name, position, size, wcs=None, extension=0):
        if file_name is None or file_name == '':
            raise ValueError('FITS file name is required.')
        else:
            fits_data = fits.getdata(file_name, ext=extension)
            return self.cutout_from_data(data=fits_data, position=position, size=size, wcs=wcs)
