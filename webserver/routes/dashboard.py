import base64
import sqlite3
from threading import Thread, Lock
from flask import Blueprint, Response, request, json, make_response
from ..extensions import socketio

bp_prefix = '/dashboard'
dashboard = Blueprint('dashboard', __name__, url_prefix=bp_prefix)

thread = None
thread_lock = Lock()

def get_dashboard_thread():
    socketio.sleep(10)

@socketio.on('get_dashboard')
def handle_get_dashboard():
    global thread

    while thread_lock:
        if thread is None:
            thread = socketio.start_background_task(get_dashboard_thread)

