FROM tensorflow/tensorflow:latest-gpu

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . /app
COPY static /app/static

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]