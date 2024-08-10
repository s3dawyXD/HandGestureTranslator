from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)


@app.route("/")
def index():
    return render_template("video.html")


@socketio.on("stream_video")
def handle_stream(data):
    # This function receives video frames from the client
    # and broadcasts them to all connected clients
    socketio.emit("video_frame", data, broadcast=True)


if __name__ == "__main__":
    socketio.run(app)
