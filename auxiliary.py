import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def check_dir_file(path):
    '''Checks if the given file path as an argument exists.'''
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid file.")

def read_video_file(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error: Could not open the video file.")
        exit()

    amount_of_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    fps = video.get(cv2.CAP_PROP_FPS)

    return video, amount_of_frames, fps

def display_shot_transitions(video_path, transitions):

    video, amount_of_frames, fps = read_video_file(video_path)

    # https://matplotlib.org/stable/gallery/widgets/buttons.html
    fig, ax = plt.subplots(1, 3)
    fig.subplots_adjust(bottom=0.2)

    class Index:
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(transitions)
            self.show_peak(i)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(transitions)
            self.show_peak(i)
            plt.draw()

        def show_peak(self, i):
            video.set(cv2.CAP_PROP_POS_FRAMES, transitions[i]-1)
            _, frame = video.read()
            ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax[0].set_title("Previous")

            video.set(cv2.CAP_PROP_POS_FRAMES, transitions[i])
            _, frame = video.read()
            ax[1].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax[1].set_title("Transition")

            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)

            # find video timestamp
            seconds = timestamp // 1000
            minutes = seconds // 60
            rem_seconds = seconds % 60

            fig.suptitle(f"Transition frame #{transitions[i]} - {i+1}/{len(transitions)} - {int(minutes)}:{int(rem_seconds)}")

            video.set(cv2.CAP_PROP_POS_FRAMES, transitions[i]+1)
            _, frame = video.read()
            ax[2].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax[2].set_title("Next")

    callback = Index()
    axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    callback.show_peak(0)

    plt.show()

