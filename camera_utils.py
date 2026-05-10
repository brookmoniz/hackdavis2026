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
"""Shared camera and runtime utilities for headless Raspberry Pi OS Lite.

Handles:
  - Logging setup (stdout / journald compatible)
  - Graceful SIGINT / SIGTERM shutdown
  - Camera open for both USB/V4L2 devices and network stream URLs
  - Frame-to-disk saving
  - FPS tracking
"""

import logging
import os
import signal
import sys
import time

import cv2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger and return a module-level logger."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
    )
    return logging.getLogger(__name__)


log = setup_logging()


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

class ShutdownFlag:
    """Thread-safe flag set by SIGINT / SIGTERM so the main loop can exit cleanly."""

    def __init__(self) -> None:
        self.running = True
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, sig, frame) -> None:  # noqa: ANN001
        log.info('Received signal %s — shutting down.', sig)
        self.running = False


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def _is_url(source: str) -> bool:
    return source.startswith(('http://', 'https://', 'rtsp://'))


def open_camera(
    source: str,
    width: int | None = None,
    height: int | None = None,
    reconnect_attempts: int = 5,
    reconnect_delay: float = 3.0,
) -> cv2.VideoCapture:
    """Open a camera from a device index or a network stream URL.

    For USB / V4L2 devices pass the device index as a string, e.g. '0'.
    For network streams pass the full URL, e.g. 'http://192.168.1.x:4747/video'.

    Resolution is set only for local V4L2 devices; for network streams
    configure resolution inside the phone app instead.

    Raises SystemExit if the camera cannot be opened after *reconnect_attempts*.
    """
    is_network = _is_url(source)

    for attempt in range(1, reconnect_attempts + 1):
        if is_network:
            log.info('Connecting to stream: %s (attempt %d/%d)', source, attempt, reconnect_attempts)
            cap = cv2.VideoCapture(source)
        else:
            device_index = int(source)
            log.info('Opening V4L2 device %d (attempt %d/%d)', device_index, attempt, reconnect_attempts)
            cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            if width and height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if cap.isOpened():
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info('Camera ready: %dx%d', actual_w, actual_h)
            return cap

        log.warning('Failed to open camera. Retrying in %.1fs...', reconnect_delay)
        time.sleep(reconnect_delay)

    log.error(
        'Could not open camera after %d attempts.\n'
        '  USB/V4L2: check `v4l2-ctl --list-devices` and ensure the driver is loaded.\n'
        '  Network:  check the phone app is running and the URL is correct.',
        reconnect_attempts,
    )
    sys.exit(1)


def read_frame(cap: cv2.VideoCapture, flip: bool = True):
    """Read one frame from *cap*, optionally mirroring horizontally.

    Returns (True, frame) on success, (False, None) on failure.
    """
    success, frame = cap.read()
    if not success:
        return False, None
    if flip:
        frame = cv2.flip(frame, 1)
    return True, frame


# ---------------------------------------------------------------------------
# FPS tracking
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling FPS counter updated every *avg_frames* frames."""

    def __init__(self, avg_frames: int = 10) -> None:
        self.avg_frames = avg_frames
        self.fps = 0.0
        self._counter = 0
        self._start = time.time()

    def update(self) -> float:
        """Call once per inference callback. Returns current FPS."""
        if self._counter % self.avg_frames == 0:
            self.fps = self.avg_frames / (time.time() - self._start)
            self._start = time.time()
        self._counter += 1
        return self.fps


# ---------------------------------------------------------------------------
# Frame saving
# ---------------------------------------------------------------------------

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    log.info('Output directory: %s', os.path.abspath(path))


def maybe_save_frame(
    frame,
    frame_index: int,
    save_every: int,
    output_dir: str,
) -> None:
    """Write *frame* to *output_dir* if this is a save frame, else do nothing."""
    if save_every <= 0 or frame is None:
        return
    if frame_index % save_every == 0:
        filename = os.path.join(output_dir, 'frame_{:08d}.jpg'.format(frame_index))
        cv2.imwrite(filename, frame)
        log.debug('Saved %s', filename)
