import os
from flask import Flask

from .extensions import socketio

from .routes.dashboard import dashboard
from .routes.streams import streams

def create_server():
    app = Flask(__name__)
    app.config['DEBUG'] = os.environ.get('PRODUCTION') != 'True'
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

    app.register_blueprint(dashboard)
    app.register_blueprint(streams)

    socketio.init_app(app)

    return app