FROM python:3.9

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . /app
COPY static /app/static

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]