import numpy as np
import cv2
import argparse
from slam_pytorch import stereo_camera_Rt


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, help='Text file with line names and transcripts for training.')
    parser.add_argument('-s', '--start-frame', required=True, help='Start from frame number.', type=int)
    parser.add_argument('-e', '--end-frame', required=True, help='End at frame number.', type=int)

    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    cap = cv2.VideoCapture(args.video)
    if cap.isOpened() == False:
        print(f'Error opening video stream or file "{args.video}".')
        exit(-1)

    lk_params = dict(winSize=(21, 21),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    orig_frame = None
    original_corners = None
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f'Failed to read frame {frame_counter} from video file "{args.video}".')
            exit(-1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_corners = None

        if frame_counter == args.start_frame:
            new_corners = cv2.goodFeaturesToTrack(gray_frame, 200, 0.002, 25)
            original_corners = np.copy(new_corners)
            orig_frame = frame

        if args.start_frame < frame_counter <= args.end_frame:
            new_corners, st, err = cv2.calcOpticalFlowPyrLK(last_gray, gray_frame, last_corners, None, **lk_params)
            st = st == 1
            st = st[:, 0]
            new_corners = new_corners[st]
            original_corners = original_corners[st]

        new_frame_show = np.copy(frame)
        if new_corners is not None:
            for i in new_corners:
                x, y = i.ravel()
                cv2.circle(new_frame_show, (x, y), 3, 255, -1)

        if original_corners is not None:
            orig_frame_show = np.copy(orig_frame)
            for i in original_corners:
                x         , y = i.ravel()
                cv2.circle(orig_frame_show, (x, y), 3, 255, -1)
        
        #    cv2.imshow('Orig frame', orig_frame_show)
        #cv2.imshow('Current frame', new_frame_show)
        #key = cv2.waitKey(1)
        #if key == ord('x'):
        #    break

        last_corners = new_corners
        last_gray = gray_frame
        frame_counter += 1

        if frame_counter > args.end_frame:
            break

    stereo_camera_Rt(original_corners, last_corners)



if __name__ == '__main__':
    main()