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
"""Image classification — headless Raspberry Pi OS Lite."""

import argparse
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera_utils import (
    ShutdownFlag, FPSCounter,
    ensure_output_dir, maybe_save_frame,
    open_camera, read_frame, log,
)


# Label panel appearance (drawn onto saved frames)
_LABEL_BG       = (255, 255, 255)  # white
_LABEL_FG       = (0, 0, 0)        # black
_LABEL_FONT     = cv2.FONT_HERSHEY_DUPLEX
_LABEL_SCALE    = 1
_LABEL_THICK    = 2
_LABEL_STEP     = 56   # pixels between label rows (rect_size 16 + margin 40)
_LABEL_PAD_W    = 600  # extra width added for the label panel


def _annotate(image, categories, fps_val: float):
    """Overlay FPS text and a right-side label panel onto *image*."""
    # FPS overlay
    cv2.putText(image, 'FPS = {:.1f}'.format(fps_val),
                (24, 50), _LABEL_FONT, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # Expand canvas for labels
    annotated = cv2.copyMakeBorder(
        image, 0, 0, 0, _LABEL_PAD_W,
        cv2.BORDER_CONSTANT, None, _LABEL_BG,
    )

    legend_x = image.shape[1] + 40 + 16  # label_margin + label_rect_size
    legend_y = image.shape[0] // 50 + 40  # matches original label_width=50

    for category in categories:
        text = '{} ({:.2f})'.format(category.category_name, category.score)
        cv2.putText(annotated, text, (legend_x, legend_y + 40),
                    _LABEL_FONT, _LABEL_SCALE, _LABEL_FG, _LABEL_THICK, cv2.LINE_AA)
        legend_y += _LABEL_STEP

    return annotated


def run(
    model: str,
    max_results: int,
    score_threshold: float,
    source: str,
    width: int,
    height: int,
    output_dir: str,
    save_every: int,
) -> None:
    shutdown = ShutdownFlag()
    fps = FPSCounter()

    if save_every > 0:
        ensure_output_dir(output_dir)

    cap = open_camera(source, width, height)

    result_list = []

    def on_result(result: vision.ImageClassifierResult, _image: mp.Image, _ts: int) -> None:
        fps.update()
        result_list.append(result)

    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ImageClassifierOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=on_result,
    )
    classifier = vision.ImageClassifier.create_from_options(options)
    log.info('Image classification model loaded: %s', model)

    frame_index = 0
    last_annotated = None

    while shutdown.running and cap.isOpened():
        ok, image = read_frame(cap)
        if not ok:
            log.error('Failed to read frame — check camera connection.')
            break

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        classifier.classify_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            time.time_ns() // 1_000_000,
        )

        if result_list:
            result = result_list.pop(0)
            categories = (result.classifications[0].categories
                          if result.classifications else [])

            if categories:
                label_strs = ['{} ({:.2f})'.format(c.category_name, c.score)
                              for c in categories]
                log.info('FPS=%.1f  Classifications: %s', fps.fps, ', '.join(label_strs))

            last_annotated = _annotate(image, categories, fps.fps)

        maybe_save_frame(last_annotated, frame_index, save_every, output_dir)
        frame_index += 1

    classifier.close()
    cap.release()
    log.info('Done. %d frames processed.', frame_index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='MediaPipe image classification — headless RPi OS Lite.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', default='classifier.tflite',
                        help='Path to the TFLite image classification model.')
    parser.add_argument('--maxResults', type=int, default=5,
                        help='Max number of classification results.')
    parser.add_argument('--scoreThreshold', type=float, default=0.0,
                        help='Minimum classification score threshold.')
    parser.add_argument('--source', default='0',
                        help='Camera source: device index (e.g. 0) or stream URL '
                             '(e.g. http://192.168.1.x:4747/video).')
    parser.add_argument('--frameWidth', type=int, default=640,
                        help='Frame width (V4L2 devices only).')
    parser.add_argument('--frameHeight', type=int, default=480,
                        help='Frame height (V4L2 devices only).')
    parser.add_argument('--outputDir', default='output_classify',
                        help='Directory to save annotated frames.')
    parser.add_argument('--saveEvery', type=int, default=30,
                        help='Save a frame every N frames. 0 = log only.')
    args = parser.parse_args()

    run(
        model=args.model,
        max_results=args.maxResults,
        score_threshold=args.scoreThreshold,
        source=args.source,
        width=args.frameWidth,
        height=args.frameHeight,
        output_dir=args.outputDir,
        save_every=args.saveEvery,
    )


if __name__ == '__main__':
    main()
