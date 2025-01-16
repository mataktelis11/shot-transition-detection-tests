# Shot transition detection

This is a collection of python scripts that perform automatic detection of transitions between shots in a video.

## Inspiration
Detecting shot transitions can be used widely, for example assisting in a video edditing software or even help streaming services to place ads during the transition so the viewer won't click away!

If you are interested in this topic you can check out the following links:

## Disclaimer
This is purely an experimental project. I will not be performing extensive evaluation tests. The proposed methods are not advanced in any way. I will be updating this in the future.

## Requirements
Developped and tested with Python 3.13.

Clone repo and create a virtual enviroment:
```
git clone https://github.com/mataktelis11/shot-transition-detection-tests.git
cd shot-transition-detection-tests
python -m venv .env
```

You will also need `ffmpeg`. If you are runnig GNU\Linux you probably have it installed already, otherwise you can get it with you package manager. For other platforms you can download it from [ffmpeg.org](https://www.ffmpeg.org/).


## Method 1: comparing color histograms of successive frames

As the title suggests, we simply compute the color histograms of each frame (one for each color channel). We then calculate the cosine distance between the histograms of successive frames.

We now have three distances for each pair of frames. To simplify things we the the average of the three distances and work with it from now on.

The vector containing the average distances can now be analyzed to obtain possible shot transitions. This is done by finding the peaks in the vector. We use scipy's `find_peaks` function with the `prominence` arguement. See more [here](https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.signal.find_peaks.html). The prominence needs to be adjusted in order to obtain results.

The script works in two steps:

1. Read the video file and create the vector with the average distances. The vector is stored as a numpy array (.npy)

2. Find the peaks in the vector. A value for the `prominence` needs to be provided. You may need to test this multiple times with different values.

### Notes
- This is a very straightforward method overall. Keep in mind that a video with high resolution will take a lot of time in order to calculate the distances. 
- Finding the transitions when the distances are already calculated is realtively fast, so it is ok to perform this multiple times with different values of `prominence`.


## Method 2: Frame Differencing and entropy
In motion detection and video compretion we often examine the differences between frames. This is called frame differencing and we can use it for transition detection.

By using `ffmpeg` we can generate a video conisting only by the frame differences. This can be called a **motion video**. By examinig the motion video we could obtain infromation about possible transitions.

When a transition occurs there should be plenty of details in the motion video. We can test this concept by calculating the **entropy** of each motion frame. 

Use the following command provided by this article: https://www.arj.no/2022/01/09/frame-differencing-with-ffmpeg

```
ffmpeg -i input.mp4 -filter_complex "format=gbrp,tblend=all_mode=difference" output.mp4
```
This will generate the motion video. Use [vlc](https://www.videolan.org/vlc/) if your video player can't open it.

