import logging
from .core import Cutout

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


class FITSCutout(Cutout):

    def __init__(self):
        super(FITSCutout, self).__init__()

    def cutout(self, file_name, position, size, extension=0):
        if file_name is None or file_name == '':
            raise ValueError('FITS file name is required.')
        else:
            with fits.open(file_name) as fits_data:
                hdu = fits_data[extension]

                # Assume SkyCoord means WCS cutout, so include it in the results
                if isinstance(position, SkyCoord):
                    wcs = WCS(hdu.header)
                    logging.info('\n\nNaxis: {}\n\n'.format(wcs.naxis))
                else:
                    wcs = None

                return self.cutout_from_data(data=hdu.data, position=position, size=size, wcs=wcs)
