FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get upgrade -y &&\
    apt-get install -y openssl wget libgl1-mesa-dev libglib2.0-0

ENV PATH /opt/conda/bin:$PATH
RUN cd /home && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -u -p /opt/conda && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda init zsh && \
    cd ~ && \
    conda install -y -c conda-forge python=3.11

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install scikit-learn scikit-learn-intelex scikit-image imageio && \
    pip install opencv-python opencv-contrib-python && \
    pip install eigen einops timm && \
    pip install tensorboard tensorboardx tqdm torchmetrics kornia && \
    pip install openpyxl yacs && \
    pip install pandas && \
    pip install matplotlib seaborn

# https://stackoverflow.com/questions/76216778/userwarning-applied-workaround-for-cudnn-issue-install-nvrtc-so
RUN cd /opt/conda/lib/python3.11/site-packages/torch/lib && \
    ln -s libnvrtc-*.so.11.2 libnvrtc.so && \
    cd ~ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
