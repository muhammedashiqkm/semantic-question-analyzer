FROM python:3.11-slim-bookworm

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Documentation only - what actually opens a port is the compose file's
# "ports", which reads the same PORT.
EXPOSE 5009

ENV FLASK_APP=run.py
ENV FLASK_ENV=production

# A container must listen on every interface, whatever .env says for a
# development run: one that binds its own loopback cannot be reached even by
# the host that published its port.
ENV HOST=0.0.0.0

# Shell form, so ${PORT} is expanded at run time. The exec form would hand
# Gunicorn the four characters "$PORT" and it would refuse to start.
CMD gunicorn --workers 2 --bind 0.0.0.0:${PORT:-5009} run:app
