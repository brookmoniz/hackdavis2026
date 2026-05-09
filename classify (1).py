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
"""Main scripts to run image classification.

Adapted for Raspberry Pi OS Lite (headless — no display server).
Instead of showing a live window, annotated frames are written to
an output directory and classification results are logged to stdout.
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

# ---------------------------------------------------------------------------
# Logging — integrates with journald when run as a systemd service
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

# Graceful shutdown on SIGINT / SIGTERM
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
      • Logs FPS and classification labels to stdout/journald.
      • Writes an annotated JPEG to *output_dir* every *save_every* frames
        (pass 0 to disable file output entirely).

    Args:
        model:            Path to the TFLite image classification model.
        max_results:      Max number of classification results.
        score_threshold:  Minimum score threshold for results.
        camera_id:        OpenCV/V4L2 camera index (usually 0 for the Pi
                          camera; run `v4l2-ctl --list-devices` to check).
        width:            Capture frame width in pixels.
        height:           Capture frame height in pixels.
        output_dir:       Directory to write annotated frames into.
                          Created automatically if it does not exist.
        save_every:       Write an annotated JPEG every N frames.
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
    # Camera — force V4L2; GStreamer pipeline rarely available on Lite
    # ------------------------------------------------------------------
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
    # FPS overlay parameters
    # ------------------------------------------------------------------
    row_size = 50       # pixels — distance from top for FPS text
    left_margin = 24    # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Label overlay parameters (drawn onto saved frames)
    label_text_color = (0, 0, 0)
    label_background_color = (255, 255, 255)  # white
    label_font_size = 1
    label_thickness = 2
    label_rect_size = 16   # pixels
    label_margin = 40
    label_padding_width = 600  # pixels — extra width added for label panel

    classification_frame = None
    classification_result_list = []

    # ------------------------------------------------------------------
    # Callback — called from the MediaPipe worker thread
    # ------------------------------------------------------------------
    def save_result(
        result: vision.ImageClassifierResult,
        unused_output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        global FPS, COUNTER, START_TIME

        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        classification_result_list.append(result)
        COUNTER += 1

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ImageClassifierOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result,
    )
    classifier = vision.ImageClassifier.create_from_options(options)
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

        # Mirror horizontally (matches original behaviour)
        image = cv2.flip(image, 1)

        # BGR → RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        classifier.classify_async(mp_image, time.time_ns() // 1_000_000)

        # Overlay FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        cv2.putText(
            image, fps_text, (left_margin, row_size),
            cv2.FONT_HERSHEY_DUPLEX, font_size, text_color,
            font_thickness, cv2.LINE_AA,
        )

        if classification_result_list:
            result = classification_result_list[0]
            categories = result.classifications[0].categories if result.classifications else []

            # Log to stdout / journald
            if categories:
                label_strs = [
                    '{} ({:.2f})'.format(c.category_name, c.score)
                    for c in categories
                ]
                log.info('FPS=%.1f  Classifications: %s', FPS, ', '.join(label_strs))

            # Build the annotated frame (label panel on the right)
            legend_x = image.shape[1] + label_margin
            legend_y = image.shape[0] // 50 + label_margin  # matches original label_width=50

            annotated = cv2.copyMakeBorder(
                image, 0, 0, 0, label_padding_width,
                cv2.BORDER_CONSTANT, None, label_background_color,
            )

            for category in categories:
                result_text = '{} ({:.2f})'.format(category.category_name, category.score)
                label_location = (legend_x + label_rect_size + label_margin,
                                  legend_y + label_margin)
                cv2.putText(
                    annotated, result_text, label_location,
                    cv2.FONT_HERSHEY_DUPLEX, label_font_size, label_text_color,
                    label_thickness, cv2.LINE_AA,
                )
                legend_y += label_rect_size + label_margin

            classification_frame = annotated
            classification_result_list.clear()

        # Save annotated frame to disk every *save_every* frames
        if save_every > 0 and classification_frame is not None:
            if frame_index % save_every == 0:
                filename = os.path.join(
                    output_dir,
                    'frame_{:08d}.jpg'.format(frame_index),
                )
                cv2.imwrite(filename, classification_frame)
                log.debug('Saved %s', filename)

        frame_index += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    classifier.close()
    cap.release()
    log.info('Classifier closed. Captured %d frames total.', frame_index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='MediaPipe image classification — headless RPi OS Lite build.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model',
        help='Path to the TFLite image classification model.',
        required=False,
        default='classifier.tflite',
    )
    parser.add_argument(
        '--maxResults',
        help='Max number of classification results.',
        required=False,
        type=int,
        default=5,
    )
    parser.add_argument(
        '--scoreThreshold',
        help='Minimum score threshold for classification results.',
        required=False,
        type=float,
        default=0.0,
    )
    # Camera IDs are usually sequential starting from 0.
    # Run `v4l2-ctl --list-devices` on the Pi to confirm the right index.
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
        default=640,
    )
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480,
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
