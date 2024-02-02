# base image
FROM tensorflow/tensorflow:2.12.0-gpu

# user and group ID arguments
ARG USER_ID=1000
ARG GROUP_ID=1000

# add group and user
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

# copy requirements
WORKDIR /tmp
COPY requirements.txt /tmp

# install requirements
RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt
