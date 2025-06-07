import cv2
"""Simple dual camera viewer."""
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Display two USB cameras side by side.")
    parser.add_argument('--cam0', type=int, default=0, help='Index of the first camera (default: 0)')
    parser.add_argument('--cam1', type=int, default=1, help='Index of the second camera (default: 1)')
    parser.add_argument('--width', type=int, default=640, help='Width of each camera frame (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Height of each camera frame (default: 480)')
    parser.add_argument(
        '--backend',
        type=str,
        default=None,
        help='OpenCV backend name (e.g. avfoundation, v4l2, dshow)'
    )
    args = parser.parse_args()

    if args.backend:
        backend_const = getattr(cv2, f'CAP_{args.backend.upper()}', 0)
        cap0 = cv2.VideoCapture(args.cam0, backend_const)
        cap1 = cv2.VideoCapture(args.cam1, backend_const)
    else:
        cap0 = cv2.VideoCapture(args.cam0)
        cap1 = cv2.VideoCapture(args.cam1)

    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open camera {args.cam0}")
    if not cap1.isOpened():
        raise RuntimeError(f"Cannot open camera {args.cam1}")

    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("Failed to read from cameras")
            break

        frame0_resized = cv2.resize(frame0, (args.width, args.height))
        frame1_resized = cv2.resize(frame1, (args.width, args.height))
        combined = np.hstack((frame0_resized, frame1_resized))

        cv2.imshow('Dual Cameras', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
