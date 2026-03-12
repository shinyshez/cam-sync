FROM python:3.12-slim

# System deps: opencv runtime, ffmpeg (viewer tests), node
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 curl git && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (layer caching)
COPY analysis/requirements.txt analysis/requirements.txt
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r analysis/requirements.txt pytest && \
    pip install --no-cache-dir git+https://github.com/cvg/LightGlue.git

# Node deps + Playwright browser
COPY viewer/package.json viewer/package.json
RUN cd viewer && npm install && npx playwright install --with-deps chromium

# Copy source
COPY analysis/ analysis/
COPY viewer/ viewer/
COPY server/ server/

EXPOSE 8787

ENTRYPOINT ["python", "-m", "server.entrypoint"]
CMD []
