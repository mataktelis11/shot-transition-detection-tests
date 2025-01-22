import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import auxiliary as aux
from scipy.signal import find_peaks
from tqdm import tqdm


# Source: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# Dense Optical Flow in OpenCV


def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='hist_cosine_dist_sd.py',
                    description="A CLI tool performing Shot transition detection on a video file using Dense Optical Flow")

    parser.add_argument('-f', '--filename', type=aux.check_dir_file, required=True, 
                        help='Provide the path of a video file')
    
    parser.add_argument('-p', '--prominence', type=float, default=0.5,
                        help='Prominence value for find_peaks function. Default value is 0.5')

    return parser.parse_args()


def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/100)
    height = int(frame.shape[0] * percent/100)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_NEAREST)


def main(video_path, prominence_value):

    video, amount_of_frames, fps = aux.read_video_file(video_path)

    all_mags = np.zeros(int(amount_of_frames)-1)
    index = 0
    pbar = tqdm(total=amount_of_frames-1)

    ret, frame1 = video.read()
    rescaled_frame = rescale_frame(frame1)
    prvs = cv.cvtColor(rescaled_frame, cv.COLOR_BGR2GRAY)

    while True:
        ret, frame2 = video.read()
        if not ret:
            pbar.close()
            print('No frames grabbed!')
            break
        rescaled_frame = rescale_frame(frame2)
        next = cv.cvtColor(rescaled_frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        avg_magnitude = np.mean(mag)

        all_mags[index] = avg_magnitude
        index+=1
        pbar.update(1)
        

        prvs = next

    # normalize to 0-1
    all_mags_norm = (all_mags-np.min(all_mags))/(np.max(all_mags)-np.min(all_mags))

    peaks, _ = find_peaks(all_mags_norm, prominence=prominence_value)

    plt.figure(figsize=(17,7))
    plt.plot(all_mags_norm)
    plt.plot(peaks, all_mags_norm[peaks], "x")

    aux.display_shot_transitions(video_path, peaks)

    plt.show()


if __name__ == "__main__":

    args = parse_arguments()

    main(args.filename, args.prominence)