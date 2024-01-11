import os
import time
import base64
import requests
import sqlite3
import numpy as np
import cv2
from threading import Thread, Lock
from flask import Blueprint, Response, request, json, make_response, send_file
from ..extensions import socketio
from ultralytics import YOLO

bp_prefix = '/streams'
streams = Blueprint('streams', __name__, url_prefix=bp_prefix)
session = requests.Session()

camera_webserver = os.environ.get('CAMERA_WEBSERVER')
port = os.environ.get('PORT')

model = YOLO('model/yolov8/yolov8n.onnx', task='detect')

thread = None
thread_lock = Lock()

def apply_bounding_boxes(results, image, threshold=0.5):
    total_objs = 0
    detected_labels = {}
    labels_accuracy = {}
    new_image = image

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score >= threshold:
            label = results.names[int(class_id)].upper()

            cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(
                new_image, 
                results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA
            )

            if label in detected_labels:
                detected_labels[label] += 1
                labels_accuracy[label].append(score)
            else:
                detected_labels[label] = 1
                labels_accuracy[label] = [score]

            total_objs += 1
            
    
    return total_objs, detected_labels, labels_accuracy, new_image

def insert_detected_labels(detected_labels, labels_accuracy):
    bulk_insert = [
        (label, round(np.average(labels_accuracy[label]), 2), detected_labels[label]) 
        for label in detected_labels
    ]

    conn = sqlite3.connect('db/webserver.db')
    cursor = conn.cursor()

    cursor.executemany(
        'INSERT INTO `detection_history` (label, accuracy, count) VALUES (?, ?, ?)',
        bulk_insert
    )

    conn.commit()
    conn.close()

def create_frame(camera_id, image):
    path = f'streams/frames/{camera_id}'

    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/{int(time.time() * 1000)}.jpeg', 'wb') as file:
        file.write(image)

def get_stream_thread(camera_id):
    while True:
        start = time.time()

        camera_port = int(port) + camera_id

        raw_image = None
        
        retry = 0
        while (retry < 3):
            try:
                frame = session.get(fr'http://{camera_webserver}:{camera_port}/jpg', timeout=10).content
            except:
                raw_image = None
                print(f'Could not retrieve image from http://{camera_webserver}:{camera_port}. Retrying for {retry + 1} time(s)...')
            
            retry += 1

        if raw_image == None:
            message = f'Could not retrieve stream from Camera {camera_id}.'

            print(message)

            socketio.emit(f'error_stream_{camera_id}', {'message': message})

            thread = None

            break

        print('Image type:', type(raw_image))

        break

        image_bytes = np.asarray(raw_image, dtype=np.uint8)
        image = cv2.imread(image_bytes, cv2.IMREAD_COLOR)

        image = cv2.convertScaleAbs(image, alpha=1.15, beta=1.15)

        results = model.predict(image, verbose=False)

        total_objs, detected_labels, labels_accuracy, new_image = apply_bounding_boxes(results[0], image)
        
        try:
            _, buffer = cv2.imencode('.jpeg', new_image)
        except:
            message = 'File could not be converted into an image.'
            
            socketio.emit(f'error_stream_{camera_id}', {'message': message})

            thread = None

            break

        # if total_objs > 0:
        #     Thread(target=insert_detected_labels, args=(detected_labels, labels_accuracy)).start()

        # if request.args.get('record', default='false') == 'true':
        #     Thread(target=create_frame, args=(camera_id, new_image)).start()

        b64_image = str(base64.b64encode(buffer), encoding='utf-8')

        data = json.dumps({
            'data': {
                'total_objects': total_objs,
                'labels': detected_labels,
                'latency': time.time() - start
            },
            'image': b64_image
        })

        print(time.time() - start)

        socketio.emit(f'get_stream_{camera_id}', data)

        socketio.sleep(0.03)

@socketio.on('get_stream')
def handle_get_stream(camera_id):
    global thread

    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(get_stream_thread, camera_id)

    socketio.emit(
        f'connected_stream_{camera_id}', 
        {'message': f'Now receiving data from Camera {camera_id}.'}
    )
