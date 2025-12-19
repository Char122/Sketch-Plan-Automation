Non-adaptive thresholding is similiar to the code in apple maps.

The adaptive thresholding file also includes cleaning functions. The tunable parameters are fairly self-explanatory and set the limits for what contours the code will consider as lanes.

The first part of the code computes the intensity gradient. It then scans through and detects edges which are detected as intensity going from low -> high -> low. This is done in both the x and y axis. The edge detection range for road markings is given by the min and max width variables. For detected edges, it sets the colour to white.

There's some morphing done as a noise reduction measure but it doesnt work very well. Kernel is currently set to 1,1 which does not do anything because 3,3 is the next smallest option but leads to a lot of blurring of the image.

The program then looks for contours. But due to the low resolution of the images, the contours are inaccurate.

Its supposed to remove contours that are too squarish and dont look like lines, currently its set to only remove perfect squares since tuning this is finnicky.

The next block of contour code removes darker blobs which are supposed to be cars -- also doesnt work very well.




