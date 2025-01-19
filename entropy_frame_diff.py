import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.stats import entropy
import auxiliary as aux

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='hist_cosine_dist_sd.py',
                    description="A CLI tool performing Shot transition detection on a video file using the entropy of the motion video")

    parser.add_argument('-f', '--filename', type=aux.check_dir_file, required=True, 
                        help='Provide the path of the original video file')

    parser.add_argument('-m', '--motionvideo', type=aux.check_dir_file, required=True, 
                        help='Provide the path of the corresponding motion video file')

    parser.add_argument('-p', '--prominence', type=float, default=6,
                        help='Prominence value for find_peaks function. Default value is 6')

    return parser.parse_args()

def calculate_entropy(video_path, output_path):

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
    np.save(output_path, average_entropy_vec)

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

    args = parse_arguments()

    # read diff video and calculate entropy of each frame
    calculate_entropy(args.motionvideo, "entropies.npy")

    find_shot_transitions(args.filename, "entropies.npy", args.prominence)