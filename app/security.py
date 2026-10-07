"""
Who may call this service.

A fixed key sent as X-API-Key, checked on every endpoint that costs money or
reaches a model. It replaces a /login that minted a JWT from an admin username
and password: the only caller is the portal, and the password had to live in
the browser's JavaScript for it to sign in - readable by anyone who opened the
site, and enough to mint a token of their own.

A key the portal keeps server-side cannot leak that way. There is no session,
nothing to expire and nothing to refresh.

An empty API_KEY enforces nothing, so this service and the portal can be
deployed in either order without a window where question checking fails. Set
it once both sides have the key.
"""

import hmac
import logging
from functools import wraps

from flask import current_app, jsonify, request

HEADER = "X-API-Key"


def require_api_key(view):
    """Refuses a request that does not carry the configured key."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        expected = (current_app.config.get("API_KEY") or "").strip()
        if not expected:
            return view(*args, **kwargs)

        # compare_digest rather than ==: a plain comparison stops at the first
        # wrong character, and the time it takes tells an attacker how much of
        # the key they have guessed.
        supplied = (request.headers.get(HEADER) or "").strip()
        if supplied and hmac.compare_digest(supplied, expected):
            return view(*args, **kwargs)

        logging.warning("Refused a request with no valid %s", HEADER)
        return jsonify({"error": "Unauthorized", "message": "Invalid or missing API key"}), 401

    return wrapped
