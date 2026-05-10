# MediaPipe Object Detection and Image Classification on Raspberry Pi OS Lite

This project uses [MediaPipe](https://github.com/google/mediapipe) with Python on a Raspberry Pi running **Raspberry Pi OS Lite** (headless, no display) to perform real-time object detection and image classification using a live stream from a mobile phone camera.

Since there is no display, results are logged to stdout and annotated frames are periodically saved to disk.

## Project structure

```
project/
├── camera_utils.py      # Shared camera, logging, FPS, and file-saving utilities
├── detect.py            # Object detection script
├── classify.py          # Image classification script
├── utils.py             # MediaPipe visualisation helper (bounding boxes)
├── efficientdet.tflite  # Object detection model
└── classifier.tflite    # Image classification model
```

## Set up your hardware

[Set up your Raspberry Pi](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up) with **Raspberry Pi OS Lite (64-bit)**.

No Pi Camera or USB camera is required. Instead, a mobile phone streams video to the Pi over Wi-Fi using an app such as:

- **IP Webcam** (Android) — streams to `http://<phone-ip>:8080/video`
- **DroidCam** (Android / iOS) — streams to `http://<phone-ip>:4747/video`

Make sure the Pi and the phone are on the same Wi-Fi network. Note the stream URL shown in the app — you will pass it as `--source` when running the scripts.

## Install dependencies

Run the setup script to install required packages and download the TFLite models:

```bash
cd mediapipe/examples/raspberry_pi
sh setup.sh
```

Or install manually:

```bash
pip install mediapipe opencv-python-headless
```

> Use `opencv-python-headless` instead of `opencv-python` — the headless build omits GUI dependencies that are unavailable on RPi OS Lite.

## Run the scripts

Both scripts can be run at the same time in separate terminals (or as separate systemd services). They share the same camera source and the same arguments.

**Object detection:**
```bash
python3 detect.py --source http://192.168.1.x:4747/video
```

**Image classification:**
```bash
python3 classify.py --source http://192.168.1.x:4747/video
```

Replace `192.168.1.x` with your phone's actual IP address as shown in the streaming app.

## Output

Each script logs detections/classifications and FPS to stdout:

```
12:00:01  INFO      Camera ready: 1280x720
12:00:01  INFO      Object detection model loaded: efficientdet.tflite
12:00:03  INFO      FPS=12.4  Detections: person (92%), cup (81%)
```

Annotated frames are saved as numbered JPEGs in the output directory every N frames:

```
output_detect/frame_00000030.jpg
output_detect/frame_00000060.jpg
...
```

## Arguments

All arguments are the same for both `detect.py` and `classify.py` unless noted.

| Argument | Description | Default |
|---|---|---|
| `--source` | Camera source: device index (e.g. `0`) or stream URL (e.g. `http://192.168.1.x:4747/video`) | `0` |
| `--model` | Path to the TFLite model file | `efficientdet.tflite` / `classifier.tflite` |
| `--maxResults` | Maximum number of results to return | `5` |
| `--scoreThreshold` | Minimum score to include a result | `0.25` (detect) / `0.0` (classify) |
| `--frameWidth` | Capture width in pixels (V4L2/USB only, ignored for streams) | `1280` (detect) / `640` (classify) |
| `--frameHeight` | Capture height in pixels (V4L2/USB only, ignored for streams) | `720` (detect) / `480` (classify) |
| `--outputDir` | Directory to save annotated frames | `output_detect` / `output_classify` |
| `--saveEvery` | Save an annotated frame every N frames. Set to `0` for log-only mode | `30` |

### Example with all arguments

```bash
python3 detect.py \
  --source http://192.168.1.x:4747/video \
  --model efficientdet.tflite \
  --maxResults 5 \
  --scoreThreshold 0.3 \
  --outputDir output_detect \
  --saveEvery 30
```

```bash
python3 classify.py \
  --source http://192.168.1.x:4747/video \
  --model classifier.tflite \
  --maxResults 5 \
  --scoreThreshold 0.1 \
  --outputDir output_classify \
  --saveEvery 30
```

## Supported models

**Object detection** (`detect.py`):
- Models from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/object_detector/index#models)
- Models trained with [MediaPipe Model Maker](https://developers.google.com/mediapipe/solutions/customization/object_detector)
- Default: `efficientdet.tflite`

**Image classification** (`classify.py`):
- Models from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/image_classifier/index#models)
- Default: `classifier.tflite`

All models must include MediaPipe metadata.

## Stopping the scripts

Press `Ctrl+C` in the terminal, or send `SIGTERM` if running as a service. Both scripts handle shutdown cleanly — the camera and model are released before the process exits.
