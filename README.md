# opencadc_cutout
Cutout library written in Python that uses astropy APIs.

## Testing

### Docker 
The easiest thing to do is to run it with docker.  OpenCADC has an [AstroQuery docker image](https://hub.docker.com/r/opencadc/astroquery/) available for that
in both Python 2.7 and Python 3.7.

#### Run tests in Docker

You can mount the local dev directory to the image and run the python test that way.  From inside the dev (working) directory:

`docker run --rm -v $(pwd):/usr/src/app opencadc/astroquery:3.7-alpine python setup.py test`

`docker run --rm -v $(pwd):/usr/src/app opencadc/astroquery:2.7-alpine python setup.py test`
