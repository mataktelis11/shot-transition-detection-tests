import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.stats import entropy
import auxiliary as aux


def calculate_entropy(video_path):

    video, amount_of_frames, fps = aux.read_video_file(video_path)

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


# this is much faster
def calculate_entropy2(video_path):

    video, amount_of_frames, fps = aux.read_video_file(video_path)

    average_entropy_vec = np.zeros(int(amount_of_frames))

    index = 0

    pbar = tqdm(total=amount_of_frames)

    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        # if frame is read correctly ret is True
        if not ret:
            pbar.close()
            print("Can't receive frame (stream end?). Exiting ...")
            break

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

        average_entropy_vec[index] = average_entropy
        index += 1
        pbar.update(1)

    # save to file
    np.save("t.npy", average_entropy_vec)

def find_shot_transitions(video_path, vector_path, prominence_value):

    video, amount_of_frames, fps = aux.read_video_file(video_path)

    average_dist = np.load(vector_path)

    # find peaks in the average distances array
    peaks, _ = find_peaks(average_dist, prominence=prominence_value)

    if len(peaks)==0:
        print("No peaks found - Try a different value for prominence")
        exit()

    plt.figure(figsize=(17,7))
    plt.plot(average_dist)
    plt.plot(peaks, average_dist[peaks], "x")

    aux.display_shot_transitions(video_path, peaks)

    plt.show()

if __name__ == "__main__":

    # read diff video and calculate entropy of each frame
    calculate_entropy2("sample_videos/bond2_diff.mp4")
    # use find_peaks - promi
    find_shot_transitions("sample_videos/bond2.mp4", "t.npy", 5)