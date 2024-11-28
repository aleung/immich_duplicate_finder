# syntax = docker/dockerfile:1.5

FROM python:3.12

WORKDIR /immich_duplicate_finder

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt && \
    pip cache purge

COPY *.py .

EXPOSE 8501
HEALTHCHECK CMD curl -fsS http://127.0.0.1:8501 | grep -c 'title>Immich Duplicate Finder</title' || exit 1

CMD streamlit run app.py
