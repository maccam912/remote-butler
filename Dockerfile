FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y chromium curl
RUN curl -sSL https://pdm.fming.dev/install-pdm.py | python3 -
COPY . .
RUN /root/.local/bin/pdm install
RUN /root/.local/bin/pdm run playwright install
CMD /root/.local/bin/pdm run python main.py
