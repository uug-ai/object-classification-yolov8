# Ultralytics YOLO 🚀, AGPL-3.0 license
# Builds ultralytics/ultralytics:latest-cpu image on DockerHub https://hub.docker.com/r/ultralytics/ultralytics
# Image is CPU-optimized for ONNX, OpenVINO and PyTorch YOLOv8 deployments

# Use the official Python 3.10 slim-bookworm as base image
FROM python:3.10-slim-bookworm

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
RUN apt update \
    && apt install --no-install-recommends -y python3-pip git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0

# Create working directory
WORKDIR /usr/src/ultralytics

# Copy contents
# COPY . /usr/src/ultralytics  # git permission issues inside container
RUN git clone https://github.com/ultralytics/ultralytics -b main /usr/src/ultralytics
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt /usr/src/ultralytics/

# Remove python3.11/EXTERNALLY-MANAGED or use 'pip install --break-system-packages' avoid 'externally-managed-environment' Ubuntu nightly error
# RUN rm -rf /usr/lib/python3.11/EXTERNALLY-MANAGED

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel
#RUN pip install --no-cache -e ".[export]" --extra-index-url https://download.pytorch.org/whl/cpu

# Run exports to AutoInstall packages
#RUN yolo export model=tmp/yolov8n.pt format=edgetpu imgsz=32
#RUN yolo export model=tmp/yolov8n.pt format=ncnn imgsz=32
# Requires <= Python 3.10, bug with paddlepaddle==2.5.0 https://github.com/PaddlePaddle/X2Paddle/issues/991
#RUN pip install --no-cache paddlepaddle>=2.6.0 x2paddle
# Remove exported models
#RUN rm -rf tmp

# Install pip requirements
COPY requirements.txt .
#RUN python -m pip --timeout=1000 install -r requirements.txt
#RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN python -m pip install -r requirements.txt 
# We have issues with nvidia drivers (we require cu118)
RUN apt-get install -y wget

RUN mkdir /ml
RUN mkdir /ml/data
RUN mkdir /ml/data/input
RUN mkdir /ml/data/output
WORKDIR /ml
COPY . .

# Environment variables
ENV MEDIA_SAVEPATH "/ml/data/input"

# Model parameters
ENV MODEL_NAME ""

# Queue parameters
ENV QUEUE_NAME "" 
ENV TARGET_QUEUE_NAME ""
ENV QUEUE_EXCHANGE ""
ENV QUEUE_HOST ""
ENV QUEUE_USERNAME ""
ENV QUEUE_PASSWORD ""

# Kerberos Vault parameters
ENV STORAGE_URI ""
ENV STORAGE_ACCESS_KEY ""
ENV STORAGE_SECRET_KEY ""

# Feature parameters
ENV PLOT ""

ENV SAVE_VIDEO ""
ENV OUTPUT_MEDIA_SAVEPATH "/ml/data/output/output_video.mp4"

ENV CREATE_BBOX_FRAME ""
ENV SAVE_BBOX_FRAME ""
ENV BBOX_FRAME_SAVEPATH "/ml/data/output/bbox_frame.jpg"

ENV CREATE_RETURN_JSON ""
ENV SAVE_RETURN_JSON ""
ENV RETURN_JSON_SAVEPATH "/ml/data/output/return_json.json"

ENV TIME_VERBOSE ""

ENV FIND_DOMINANT_COLORS ""

# Classification parameters
ENV CLASSIFICATION_FPS ""
ENV CLASSIFICATION_THRESHOLD ""
ENV MAX_NUMBER_OF_PREDICTIONS ""
ENV MIN_DISTANCE ""
ENV MIN_STATIC_DISTANCE ""
ENV MIN_DETECTIONS ""
ENV ALLOWED_CLASSIFICATIONS "0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28"


# Run the application
ENTRYPOINT ["python" , "object_classification_yolov8.py"]


