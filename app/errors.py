"""
One shape for every failure, and the right code with it.

WHY IT IS WORTH HAVING

A caller had three different bodies to read: marshmallow's own dict for a
validation error, {"error": "..."} for most things, and {"error", "message"}
for the key. The portal gave up and showed "An error occurred during search"
for all of them, so a teacher never learned which of these it was:

  - they named a provider this service does not have
  - the bank URL could not be fetched
  - a model was unreachable for a minute
  - something here is broken

The first two they can act on; the third means try again; only the last is
ours. They used to arrive as 500, 404 and 500 respectively, which told the
portal nothing it could act on either.

Every error now answers:

    {"error": "<short, stable, for code>", "message": "<a sentence for a person>"}

and the message names what IS available wherever it can, because "model name
not found for a specified provider" leaves an administrator nowhere to go.
"""

from typing import Any, Dict, Tuple

from flask import Response, jsonify

JsonResponse = Tuple[Response, int]


def fail(status: int, error: str, message: str) -> JsonResponse:
    """One error, in the one shape."""
    return jsonify({"error": error, "message": message}), status


def bad_request(message: str) -> JsonResponse:
    """The request asks for something impossible. Changing it would help."""
    return fail(400, "Bad Request", message)


def upstream_failed(message: str) -> JsonResponse:
    """Something this service depends on answered badly. Not the caller's doing."""
    return fail(502, "Bad Gateway", message)


def unavailable(message: str) -> JsonResponse:
    """A provider is unreachable or unconfigured here. Worth retrying later."""
    return fail(503, "Service Unavailable", message)


def internal(message: str = "An internal server error occurred.") -> JsonResponse:
    """A bug here. Logged with a traceback; the caller is told nothing else."""
    return fail(500, "Internal Server Error", message)


def invalid_payload(messages: Dict[str, Any]) -> JsonResponse:
    """
    A marshmallow ValidationError, flattened into a sentence.

    Its own dict is kept under "fields" for anything that wants to mark up a
    form, but the message is what a person reads.
    """
    parts = []
    for field, problems in (messages or {}).items():
        if isinstance(problems, (list, tuple)):
            parts.append(f"{field}: {'; '.join(str(p) for p in problems)}")
        else:
            parts.append(f"{field}: {problems}")

    body = {
        "error": "Bad Request",
        "message": "The request is missing something: " + ", ".join(parts)
        if parts else "The request could not be read.",
        "fields": messages,
    }
    return jsonify(body), 400
