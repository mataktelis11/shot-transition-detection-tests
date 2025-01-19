import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
import auxiliary as aux


def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='hist_cosine_dist_sd.py',
                    description="A CLI tool performing Shot transition detection on a video file using the cosine distance of the frames' color histograms")

    parser.add_argument('-f', '--filename', type=aux.check_dir_file, required=True, 
                        help='Provide the path of a video file')
    
    parser.add_argument('-p', '--prominence', type=float, default=0.5,
                        help='Prominence value for find_peaks function. Default value is 0.5')

    return parser.parse_args()

def calculate_distances(video_path, output_path):

    video, amount_of_frames, fps = aux.read_video_file(video_path)

    # initialize vectors for distances
    dist_r = np.zeros(int(amount_of_frames)-1)
    dist_g = np.zeros(int(amount_of_frames)-1)
    dist_b = np.zeros(int(amount_of_frames)-1)

    # read the first frame
    ret, current_frame = video.read()
    # get color histograms
    hist_curr_r = cv2.calcHist([current_frame],[0],None,[256],[0,256])
    hist_curr_g = cv2.calcHist([current_frame],[1],None,[256],[0,256])
    hist_curr_b = cv2.calcHist([current_frame],[2],None,[256],[0,256])

    index = 0
    pbar = tqdm(total=amount_of_frames-1)

    while True:
        # Capture frame-by-frame
        ret, next_frame = video.read()
        # if frame is read correctly ret is True
        if not ret:
            pbar.close()
            print("Can't receive frame (stream end?). Exiting ...")
            break

        hist_next_r = cv2.calcHist([next_frame],[0],None,[256],[0,256])
        hist_next_g = cv2.calcHist([next_frame],[1],None,[256],[0,256])
        hist_next_b = cv2.calcHist([next_frame],[2],None,[256],[0,256])

        # calculate cosine distance
        dist_r[index] = 1 - (np.dot(hist_curr_r.T, hist_next_r)/(np.linalg.norm(hist_curr_r)*np.linalg.norm(hist_next_r)))[0,0]
        dist_g[index] = 1 - (np.dot(hist_curr_g.T, hist_next_r)/(np.linalg.norm(hist_curr_r)*np.linalg.norm(hist_next_r)))[0,0]
        dist_b[index] = 1 - (np.dot(hist_curr_b.T, hist_next_r)/(np.linalg.norm(hist_curr_r)*np.linalg.norm(hist_next_r)))[0,0]

        # store histograms for next iteration
        hist_curr_r = hist_next_r
        hist_curr_g = hist_next_g
        hist_curr_b = hist_next_b

        index += 1
        pbar.update(1)

    # get average values of color channels
    average_dist = np.zeros(int(amount_of_frames)-1)
    average_dist = (dist_r + dist_g + dist_b) / 3

    # save to file
    np.save(output_path, average_dist)

def find_shot_transitions(video_path, vector_path, prominence_value):

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

    calculate_distances(args.filename, "average_dist.npy")

    find_shot_transitions(args.filename, "average_dist.npy", args.prominence)

