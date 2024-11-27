# syntax = docker/dockerfile:1.5

FROM python:3.12

RUN git clone https://github.com/vale46n1/immich_duplicate_finder.git /immich_duplicate_finder

WORKDIR /immich_duplicate_finder

RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt && \
    pip cache purge

EXPOSE 8501
HEALTHCHECK CMD curl -fsS http://127.0.0.1:8501 | grep -c 'title>Immich Duplicate Finder</title' || exit 1

CMD streamlit run app.py
