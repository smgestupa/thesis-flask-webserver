from webserver import create_server, socketio

server = create_server()
socketio.run(server)