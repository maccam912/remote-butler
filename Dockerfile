FROM python:3.10-slim

RUN apt-get update && apt-get install -y chromium
RUN curl -sSL https://pdm.fming.dev/dev/install-pdm.py | python3 -
COPY . .
RUN /root/.local/bin/pdm install
RUN /root/.local/bin/pdm run playwright install
CMD /root/.local/bin/pdm run python main.py
