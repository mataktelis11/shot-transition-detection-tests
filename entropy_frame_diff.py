import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.stats import entropy

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

def calculate_entropy(video_path):

    video, amount_of_frames, fps = read_video_file(video_path)

    average_entropy_vec = np.zeros(int(amount_of_frames))

    # use tqdm progress bar
    for i in tqdm(range(int(amount_of_frames))):

        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = video.read()
        
        # https://unimatrixz.com/blog/latent-space-image-quality-with-entropy/

        # get histogram of each color channel
        hist_r = cv2.calcHist([frame],[0],None,[256],[0,256])
        hist_g = cv2.calcHist([frame],[1],None,[256],[0,256])
        hist_b = cv2.calcHist([frame],[2],None,[256],[0,256])
        # get entropy of each color channel
        image_entropy_r = entropy(hist_r / hist_r.sum(), base=2)
        image_entropy_g = entropy(hist_g / hist_g.sum(), base=2)
        image_entropy_b = entropy(hist_b / hist_b.sum(), base=2)
        
        average_entropy = np.average([image_entropy_r, image_entropy_g, image_entropy_b])

        average_entropy_vec[i] = average_entropy

    # save to file
    np.save("t.npy", average_entropy_vec)


def find_shot_transitions(video_path, vector_path, prominence_value):

    video, amount_of_frames, fps = read_video_file(video_path)

    average_dist = np.load(vector_path)

    # find peaks in the average distances array
    peaks, _ = find_peaks(average_dist, prominence=prominence_value)

    if len(peaks)==0:
        print("No peaks found - Try a different value for prominence")
        exit()

    plt.figure(figsize=(17,7))
    plt.plot(average_dist)
    plt.plot(peaks, average_dist[peaks], "x")

    # https://matplotlib.org/stable/gallery/widgets/buttons.html
    fig, ax = plt.subplots(1, 3)
    fig.subplots_adjust(bottom=0.2)

    class Index:
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(peaks)
            self.show_peak(i)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(peaks)
            self.show_peak(i)
            plt.draw()

        def show_peak(self, i):
            video.set(cv2.CAP_PROP_POS_FRAMES, peaks[i]-1)
            _, frame = video.read()
            ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax[0].set_title("Previous")

            video.set(cv2.CAP_PROP_POS_FRAMES, peaks[i])
            _, frame = video.read()
            ax[1].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax[1].set_title("Transition")

            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)

            # find video timestamp
            seconds = timestamp // 1000
            minutes = seconds // 60
            rem_seconds = seconds % 60

            fig.suptitle(f"Transition frame #{peaks[i]} - {i+1}/{len(peaks)} - {int(minutes)}:{int(rem_seconds)}")

            video.set(cv2.CAP_PROP_POS_FRAMES, peaks[i]+1)
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



#calculate_entropy("sample_videos/output.mp4")

find_shot_transitions("sample_videos/bond.mp4", "t.npy", 4)