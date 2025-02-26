FROM python:3.10

RUN pip install numpy && \
    pip install opencv-python

RUN apt-get update -y && \
    apt-get install -y libgl1
    
RUN git clone https://github.com/opencv/opencv_extra.git

