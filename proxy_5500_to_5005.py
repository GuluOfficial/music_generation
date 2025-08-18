from flask import Flask, request, Response
import requests, json

TARGET = "http://127.0.0.1:5005"
app = Flask(__name__)

@app.route("/healthz", methods=["GET"])
def healthz():
    r = requests.get(f"{TARGET}/healthz", timeout=5)
    return Response(r.content, r.status_code, r.headers.items())

@app.route("/synthesis_music", methods=["POST"])
def synthesis_music():
    data = request.get_json(force=True, silent=True) or {}
    r = requests.post(f"{TARGET}/synthesis_music", json=data, timeout=900)
    return Response(r.content, r.status_code, [("Content-Type", r.headers.get("Content-Type","application/json"))])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, threaded=True)
