# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run object detection.

Adapted for Raspberry Pi OS Lite (headless — no display server).
Instead of showing a live window, annotated frames are written to
an output directory and detection results are logged to stdout.
"""

import argparse
import logging
import os
import signal
import sys
import time


import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize

# ---------------------------------------------------------------------------
# Logging — replaces the implicit print-to-nowhere of the original
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()

# Graceful shutdown on SIGINT / SIGTERM (important for headless services)
_running = True


def _signal_handler(sig, frame):  # noqa: ANN001
    global _running
    log.info('Received signal %s — shutting down.', sig)
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run(
    model: str,
    max_results: int,
    score_threshold: float,
    camera_id: int,
    width: int,
    height: int,
    output_dir: str,
    save_every: int,
) -> None:
    """Continuously run inference on images acquired from the camera.

    On Raspberry Pi OS Lite there is no display server, so this version:
      • Logs FPS and detection labels to stdout/journald.
      • Writes an annotated JPEG to *output_dir* every *save_every* frames
        (pass 0 to disable file output entirely).

    Args:
        model:            Path to the TFLite object detection model.
        max_results:      Max number of detection results.
        score_threshold:  Minimum score threshold for results.
        camera_id:        OpenCV camera index (usually 0 for the Pi camera
                          when using the V4L2 driver, e.g. /dev/video0).
        width:            Capture frame width in pixels.
        height:           Capture frame height in pixels.
        output_dir:       Directory to write annotated frames into.
                          Created automatically if it does not exist.
        save_every:       Write an annotated frame every N frames.
                          Set to 0 to disable saving entirely.
    """
    global _running

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if save_every > 0:
        os.makedirs(output_dir, exist_ok=True)
        log.info('Annotated frames will be saved to: %s', os.path.abspath(output_dir))

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    # Force V4L2 backend — the default GStreamer pipeline may not be
    # available on RPi OS Lite.
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        log.error(
            'Cannot open camera %d. Check that the camera is enabled '
            '(raspi-config → Interface Options → Camera) and that the '
            'V4L2 driver is loaded (sudo modprobe bcm2835-v4l2).', camera_id
        )
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info('Camera opened: %dx%d', actual_w, actual_h)

    # ------------------------------------------------------------------
    # Visualisation / FPS parameters
    # ------------------------------------------------------------------
    row_size = 50        # pixels — distance from top for FPS text
    left_margin = 24     # pixels
    text_color = (0, 0, 0)   # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_frame = None
    detection_result_list = []

    # ------------------------------------------------------------------
    # Callback — called from the MediaPipe worker thread
    # ------------------------------------------------------------------
    def save_result(
        result: vision.ObjectDetectorResult,
        unused_output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        global FPS, COUNTER, START_TIME

        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        detection_result_list.append(result)
        COUNTER += 1

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result,
    )
    detector = vision.ObjectDetector.create_from_options(options)
    log.info('Model loaded: %s', model)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    frame_index = 0

    while _running and cap.isOpened():
        success, image = cap.read()
        if not success:
            log.error(
                'Failed to read frame from camera. '
                'Check cable and camera module.'
            )
            break

        # Mirror horizontally (matches original behaviour; remove if
        # the camera is mounted in the other orientation).
        image = cv2.flip(image, 1)

        # BGR → RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Overlay FPS on the current frame
        fps_text = 'FPS = {:.1f}'.format(FPS)
        cv2.putText(
            image, fps_text, (left_margin, row_size),
            cv2.FONT_HERSHEY_DUPLEX, font_size, text_color,
            font_thickness, cv2.LINE_AA,
        )

        if detection_result_list:
            result = detection_result_list[0]

            # Log detected labels (useful when tailing journald or a log file)
            labels = [
                '{} ({:.0%})'.format(
                    d.categories[0].category_name,
                    d.categories[0].score,
                )
                for d in result.detections
                if d.categories
            ]
            if labels:
                log.info('FPS=%.1f  Detections: %s', FPS, ', '.join(labels))

            image = visualize(image, result)
            detection_frame = image
            detection_result_list.clear()

        # Save annotated frame to disk every *save_every* frames
        if save_every > 0 and detection_frame is not None:
            if frame_index % save_every == 0:
                filename = os.path.join(
                    output_dir,
                    'frame_{:08d}.jpg'.format(frame_index),
                )
                cv2.imwrite(filename, detection_frame)
                log.debug('Saved %s', filename)

        frame_index += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    detector.close()
    cap.release()
    log.info('Detector closed. Captured %d frames total.', frame_index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='MediaPipe object detection — headless RPi OS Lite build.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model',
        help='Path to the TFLite object detection model.',
        required=False,
        default='efficientdet.tflite',
    )
    parser.add_argument(
        '--maxResults',
        help='Max number of detection results.',
        required=False,
        type=int,
        default=5,
    )
    parser.add_argument(
        '--scoreThreshold',
        help='Minimum score threshold for detection results.',
        required=False,
        type=float,
        default=0.25,
    )
    # Camera IDs are usually sequential starting from 0.
    # On RPi with the official camera module and V4L2 driver, this is
    # typically 0 (/dev/video0).  Run `v4l2-ctl --list-devices` to check.
    parser.add_argument(
        '--cameraId',
        help='V4L2 camera index (see v4l2-ctl --list-devices).',
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=1280,
    )
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=720,
    )
    parser.add_argument(
        '--outputDir',
        help='Directory to save annotated frames. Created if absent.',
        required=False,
        default='output_frames',
    )
    parser.add_argument(
        '--saveEvery',
        help=(
            'Save an annotated JPEG every N frames. '
            'Set to 0 to disable file output (log-only mode).'
        ),
        required=False,
        type=int,
        default=30,
    )

    args = parser.parse_args()

    run(
        model=args.model,
        max_results=args.maxResults,
        score_threshold=args.scoreThreshold,
        camera_id=args.cameraId,
        width=args.frameWidth,
        height=args.frameHeight,
        output_dir=args.outputDir,
        save_every=args.saveEvery,
    )


if __name__ == '__main__':
    main()
