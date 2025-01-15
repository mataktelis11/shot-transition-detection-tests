import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm
from scipy.signal import find_peaks

def check_dir_file(path):
    '''Checks if the given file path as an argument exists.'''
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid file.")

def parse_arguments():

    # https://stackoverflow.com/questions/17909294/python-argparse-mutual-exclusive-group

    # create the top-level parser
    parser = argparse.ArgumentParser(
                    prog='hist_cosine_dist_sd.py',
                    description="A CLI tool performing Shot transition detection on a video file using the cosine distance of the frames' color histograms")

    subparsers = parser.add_subparsers(help="Use subcommand 'calculate_distances' or 'find_transitions'", dest="subcommand")

    # create the parser for the "calculate_distances" command
    parser_a = subparsers.add_parser('calculate_distances', help='Calculate and store the distnaces vector')
    parser_a.add_argument('-f', '--filename', type=check_dir_file, required=True, help="The video file to perform shot detection")
    parser_a.add_argument('-o', '--output', type=str, default='average_dist.npy', help="Name of file that will save the distances vector. Default is 'average_dist.npy'")

    # create the parser for the "find_transitions" command
    parser_b = subparsers.add_parser('find_transitions', help='Use the distances vector to find transitions')
    parser_b.add_argument('-f', '--filename', type=check_dir_file, required=True, help="The video file to perform shot detection")
    parser_b.add_argument('-v', '--vector_filename', type=check_dir_file, required=True, help="The file containing the distances vector")

    parser_b.add_argument('-p', '--prominence',
                          type=float,
                          default=0.5,
                          help="The prominence arguement for scipy's find_peaks function. Default value is 0.5")

    return parser.parse_args()

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


def calculate_distances(video_path, output_path):

    video, amount_of_frames, fps = read_video_file(video_path)

    # initialize vectors for distances
    dist_r = np.zeros(int(amount_of_frames)-1)
    dist_g = np.zeros(int(amount_of_frames)-1)
    dist_b = np.zeros(int(amount_of_frames)-1)

    # read the first frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, current_frame = video.read()

    hist_curr_r = cv2.calcHist([current_frame],[0],None,[256],[0,256])
    hist_curr_g = cv2.calcHist([current_frame],[1],None,[256],[0,256])
    hist_curr_b = cv2.calcHist([current_frame],[2],None,[256],[0,256])

    # use tqdm progress bar
    for i in tqdm(range(int(amount_of_frames)-1)):

        video.set(cv2.CAP_PROP_POS_FRAMES, i+1)
        _, next_frame = video.read()

        hist_next_r = cv2.calcHist([next_frame],[0],None,[256],[0,256])
        hist_next_g = cv2.calcHist([next_frame],[1],None,[256],[0,256])
        hist_next_b = cv2.calcHist([next_frame],[2],None,[256],[0,256])

        # calculate cosine distance
        dist_r[i] = 1 - (np.dot(hist_curr_r.T, hist_next_r)/(np.linalg.norm(hist_curr_r)*np.linalg.norm(hist_next_r)))[0,0]
        dist_g[i] = 1 - (np.dot(hist_curr_g.T, hist_next_r)/(np.linalg.norm(hist_curr_r)*np.linalg.norm(hist_next_r)))[0,0]
        dist_b[i] = 1 - (np.dot(hist_curr_b.T, hist_next_r)/(np.linalg.norm(hist_curr_r)*np.linalg.norm(hist_next_r)))[0,0]

        # store histograms for next iteration
        hist_curr_r = hist_next_r
        hist_curr_g = hist_next_g
        hist_curr_b = hist_next_b

    # get average values of color channels
    average_dist = np.zeros(int(amount_of_frames)-1)
    average_dist = (dist_r + dist_g + dist_b) / 3

    # save to file
    np.save(output_path, average_dist)

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

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    if args.subcommand is None:
        print("You must provide a subcommand!")
        exit()

    if args.subcommand == 'calculate_distances':
        calculate_distances(args.filename, args.output)

    if args.subcommand == 'find_transitions':
        find_shot_transitions(args.filename, args.vector_filename, args.prominence)
