# EtAlii.Dllm server image: dllm-server on port 5080 (OpenAI and Anthropic APIs, the chat page at /).
# See docs/getting-started.md#docker. Build: docker build -t etalii-dllm .
#
# The kernels are compiled in the builder stage with the flags from CMakeLists.txt; the SIMD path is picked at run
# time, so the image gives the same bits as a native install on the same machine.

# The full image has the C++ toolchain; scikit-build-core fetches CMake from PyPI.
FROM python:3.12 AS build
WORKDIR /src
COPY pyproject.toml CMakeLists.txt README.md LICENSE ./
COPY cpp cpp
COPY src src
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
# The cuda extra adds NVRTC, so `docker run --gpus all ... -e DLLM_DEVICE=cuda` works with the NVIDIA container
# toolkit; the driver comes from the host.
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir "$(ls /wheels/*.whl)[cuda]"
COPY docker/entrypoint.sh /usr/local/bin/dllm-entrypoint
RUN useradd --create-home --uid 1000 dllm && mkdir /models && chown dllm /models
USER dllm
VOLUME /models
ENV DLLM_MODEL_FILE=/models/model.dllm
EXPOSE 5080
ENTRYPOINT ["dllm-entrypoint"]
CMD ["dllm-server", "--host", "0.0.0.0", "--port", "5080"]
