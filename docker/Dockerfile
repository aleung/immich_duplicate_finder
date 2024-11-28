# syntax = docker/dockerfile:1.5

# Download stage
FROM python:3.12 AS builder

RUN git clone https://github.com/vale46n1/immich_duplicate_finder.git  /immich_duplicate_finder

# Final image stage
FROM python:3.12

WORKDIR /immich_duplicate_finder

COPY --from=builder /immich_duplicate_finder/requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt && \
    pip cache purge

COPY --from=builder /immich_duplicate_finder/*.py .

EXPOSE 8501
HEALTHCHECK CMD curl -fsS http://127.0.0.1:8501 | grep -c 'title>Immich Duplicate Finder</title' || exit 1

CMD streamlit run app.py
