# BiGuass_Fit
Simple utility to fit a bigaussian (two gaissian peaks) to spectroscopic data, with initial parameters coming from user mouse clicks. Uses the `imunuit` library to perform fits - this is a prerequisite along with the standard things (numpy/scipy/matplotlib).

Some examples showing use are given in the included Jupyter notebook. A minimal example is as follows:
```
import numpy as np
import matplotlib.pyplot as plt
from bigaus_fit import *
```
```
%matplotlib notebook
counts,edges = np.histogram(<some data>,...)
fitter = BiGausFitter(
    counts, edges,
    dofit = True,
    dohesse = True,
    title = 'My Histogram'
)
```
(Take note of the `%matplotlib notebook` call - this is required for interactive input in a notebook cell.)

This will plot the histogram on a new canvas, and the user should then do the following:
  1. click on the peak of the first Gaussian --> draws a horizontal line showing the 1/2 max line
  2. click on the point where the horizontal line intersects the histogram (left of the peak)
  3. click on the point where the horizontal line intersects the histogram (right of the peak)
  4. repeat steps 1-3, now for the second peak

After the clicks are registered, the fitter will be run, and the fit results plotted on top of the histogram data.

There are various options, e.g. turning off automatic fitting, excluding a `hesse()` call in the fitter, etc. These have not yet been well documented but are hopefully clear from context/examples. Fell free to contact the author if you have questions.
