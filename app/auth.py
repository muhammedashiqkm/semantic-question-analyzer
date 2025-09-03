import logging
from flask import request, jsonify, Blueprint, current_app
from flask_jwt_extended import create_access_token
from marshmallow import ValidationError
from .schemas import LoginSchema
from . import limiter

auth_bp = Blueprint('auth', __name__)
login_schema = LoginSchema()

@auth_bp.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    try:
        data = login_schema.load(request.get_json())
    except ValidationError as err:
        return jsonify(err.messages), 400
    
    username = data['username']
    password = data['password']

    admin_user = current_app.config['ADMIN_USERNAME']
    admin_pass = current_app.config['ADMIN_PASSWORD']

    if username == admin_user and password == admin_pass:
        logging.info(f"Successful login for user: {username}")
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)

    logging.warning(f"Failed login attempt for user: {username}")
    return jsonify({"error": "Bad username or password"}), 401