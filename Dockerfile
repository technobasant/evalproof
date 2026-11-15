# syntax=docker/dockerfile:1
#
# The evalproof command-line tool as a container image: `docker run --rm IMAGE --version`.
# The image holds Python 3.13, evalproof and its four runtime dependencies, and nothing else.
# It runs as a non-root user (uid 10001) in /work, where a suite directory is mounted.
# It has no coding-agent CLI (`claude`, `codex`, ...): a suite that targets one fails with the
# usual "command not found" error. Suites that use a Python function, an HTTP endpoint or a
# hosted model through an API key run as they do anywhere else.
#
# One wheel goes into the image, from one of two places. WHEEL_SOURCE picks it (BuildKit
# builds only the stage that is selected):
#
#   build     (default) build the wheel from this source tree:
#               docker build -t evalproof .
#   prebuilt  install the wheel(s) already in ./dist. The release workflow uses this, so the
#             image holds the exact files that were published to PyPI:
#               uv build --wheel && docker build --build-arg WHEEL_SOURCE=prebuilt \
#                 --build-arg VERSION=1.3.0 -t evalproof .
#
# Either way the final stage installs the wheel with its dependencies from PyPI, so a build
# needs the network. VERSION and REVISION only fill in the image labels.

ARG PYTHON_VERSION=3.13
ARG WHEEL_SOURCE=build

FROM python:${PYTHON_VERSION}-slim AS wheel-build
WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM scratch AS wheel-prebuilt
COPY dist/*.whl /wheels/

FROM wheel-${WHEEL_SOURCE} AS wheels

FROM python:${PYTHON_VERSION}-slim
ARG VERSION=dev
ARG REVISION=unknown

LABEL org.opencontainers.image.title="evalproof" \
      org.opencontainers.image.description="Treat LLM evaluation as measurement: calibrate the judge, correct the pass rate, gate CI on significant regressions." \
      org.opencontainers.image.source="https://github.com/technobasant/evalproof" \
      org.opencontainers.image.url="https://basantbhattarai.com.np/evalproof/" \
      org.opencontainers.image.documentation="https://basantbhattarai.com.np/evalproof/getting-started/install/" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# The wheel is mounted for this one step, so it is not kept in a layer.
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*.whl \
 && evalproof --version

RUN groupadd --gid 10001 evalproof \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin evalproof \
 && mkdir /work \
 && chown evalproof:evalproof /work

USER evalproof
WORKDIR /work
ENTRYPOINT ["evalproof"]
CMD ["--help"]
