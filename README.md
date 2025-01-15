# Shot transition detection

This is a collection of python scripts that perform automatic detection of transitions between shots in a video.

## Inspiration
Detecting shot transitions can be used widely, for example assisting in a video edditing software or even help streaming services to place ads during the transition so the viewer won't click away!

If you are interested in this topic you can check out the following links:

## Disclaimer
This is purely an experimental project. I will not be performing extensive evaluation tests.


## Set up
Developped and tested with Python 3.13.

Clone repo and create virtual enviroment:
```
git clone
cd
python 
```

## Method 1: comparing color histograms of successive frames

As the title suggests, we simply compute the color histograms of each frame (one for each color channel). We then calculate the cosine distance between the histograms of successive frames.

We now have three distances for each pair of frames. To simplify things we the the average of the three distances and work with it from now on.

The vector containing the average distances can now be analyzed to obtain possible shot transitions. This is done by finding the peaks in the vector. We use scipy's `find_peaks` function with the `prominence` arguement. See more [here](https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.signal.find_peaks.html). The prominence needs to be adjusted in order to obtain results.

The script works in two steps:

1. Read the video file and create the vector with the average distances. The vector is stored as a numpy array (.npy)

2. Find the peaks in the vector. A value for the `prominence` needs to be provided. You may need to test this multiple times with different values.