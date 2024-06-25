from flask import Flask, render_template, Response, request, redirect, url_for
from camera import VideoCamera


import cv2
from bidi.algorithm import get_display
import arabic_reshaper


test = VideoCamera()

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():

    if request.method == "POST":
        return redirect(url_for("video_feed"))
    else:
        print(request.args)
        return redirect(url_for("video_feed"))


def gen(camera):
    while True:
        frame = camera.get_frame()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen(test), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/show_text")
def show_text():

    x = test.text()

    x = "".join(x)

    return render_template("text.html", x=x)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
