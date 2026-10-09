# Versioned base tags; record resolved image digests when building a release.
FROM node:22.14.0-bookworm-slim AS node-runtime
FROM python:3.10.16-slim-bookworm
WORKDIR /app
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
COPY requirements-jury.lock ./
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements-jury.lock
COPY tools/browser/package.json tools/browser/package-lock.json tools/browser/
RUN npm ci --prefix tools/browser \
    && node tools/browser/node_modules/playwright/cli.js install --with-deps chromium
COPY . .
ARG SOURCE_REVISION
ENV SOURCE_REVISION=${SOURCE_REVISION} \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    POLARS_MAX_THREADS=1 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-B", "-X", "utf8", "run_all.py"]
