FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RSDET_SUBMISSION_CONFIG=/app/config.json

RUN sed -i 's|http://security.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g' \
        /etc/apt/sources.list \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    bash ca-certificates curl bzip2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --silent --show-error --retry 5 --retry-all-errors \
        https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

COPY environment.yml /tmp/environment.yml
RUN micromamba config set download_threads 1 \
    && micromamba config set extract_threads 1 \
    && micromamba create -y -n detector -f /tmp/environment.yml \
    && micromamba clean --all --yes

ENV PATH=/opt/micromamba/envs/detector/bin:/opt/micromamba/bin:$PATH
ENV YOLO_CONFIG_DIR=/tmp

# Ultralytics 的开放上界会在新索引中选择 OpenCV 5，并把已冻结的 NumPy 1.26
# 升到 2.x。部署镜像在重依赖层之后显式恢复经过项目验证的二进制组合。
RUN python -m pip install --no-cache-dir --force-reinstall \
        numpy==1.26.4 opencv-python==4.10.0.84 \
    && python -c "import cv2, numpy; assert cv2.__version__ == '4.10.0'; assert numpy.__version__ == '1.26.4'"

COPY app /app
COPY models /app/models

RUN mkdir -p /input /output

ENTRYPOINT ["python", "/app/main.py"]
