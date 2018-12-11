---
layout: post
title: "Java 8 streams Examples"
excerpt: "Functional Programming in java 8"
tags:
  - java 8
  - Lambda Expressions
  - Streams
last_modified_at: 2017-12-3T13:48:50+05:30
---

```python
import numpy as np

DISTANCE_SMOOTHING = 0.1


def make_gaussian(width, height, center=None, fwhm = 3):
    """ Make a square gaussian kernel.
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    y.shape = (height, 1)
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0, y0 = center
    
    return np.exp(-4 * np.log(2) * ((DISTANCE_SMOOTHING * (x-x0)) ** 2 + (DISTANCE_SMOOTHING * (y-y0)) ** 2) / fwhm ** 2)
```


