# High Dynamic Range Imaging

## Synopsis

In this project, I'm practicing on the core algorithms behind computing HDR images based on the paper “Recovering High Dynamic Range Radiance Maps from Photographs” by Debevec & Malik (available in Canvas under Files --> Papers).


## Directions

### 1. Implement the functions in the `hdr.py` file.

- `linearWeight`: Determine the weight of a pixel based on its intensity.
- `sampleIntensities`: Randomly sample pixel intensity exposure slices for each possible pixel intensity value from the exposure stack.
- `computeResponseCurve`: Find the camera response curve for a single color channel by finding the least-squares solution to an overdetermined system of equations.
- `computeRadianceMap`: Use the response curve to calculate the radiance map for each pixel in the current color layer.


*Notes*:
- Images in the `images/source/sample` directory are provided for testing. 

- Downsampling your images to 1-2 MB each will save processing time during development. (Larger images take longer to process.)

- It is essential to put your images in exposure order and name them in this order, similar to the input/sample images. For the given sample images of the home, the exposure info is given in main.py and repeated here (darkest to lightest):
`EXPOSURE TIMES = np.float64([1/160.0, 1/125.0, 1/80.0, 1/60.0, 1/40.0, 1/15.0])`

- Image alignment is critical for HDR. Whether you take your own images or use a set from the web, ensure that your images are aligned and cropped to the same dimensions. This will require a tripod or improvised support. You may use a commercial program such as Gimp (free) or Photoshop ($$) to help with this step. 


### 2. Use these function on your own input images to make an HDR image 

You MUST use at least 5 input images to create your final HDR image. In most cases, using more input images is better. We recommend resizing images to small versions (like the ones provided) so that the code runs more quickly.

Look online for a set of HDR images with exposure data (EXIF) for exposure times, apertures, and ISO settings. You will need to enter the exposure times (not exposure values) into the main.py code in the same order as the image files are numbered to get a correct HDR result. Go from darkest to lightest. Your results can be amazingly bad if you don't follow this rule.

You may take your own images if your camera supports manual exposure control (at least 5 input images). In particular, the exposure times are required, and aperture and ISO must be reported. Aperture & ISO two settings should be held constant. Dark indoor scenes with bright outdoors generally work great for this, or other scenes with overly-bright and dark areas.
