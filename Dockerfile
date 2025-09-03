FROM python:3.11-slim-bookworm

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV FLASK_APP=run.py
ENV FLASK_ENV=production

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "run:app"]
