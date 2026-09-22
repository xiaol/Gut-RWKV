from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from threading import Lock


def create_server(model, host="127.0.0.1", port=8000):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("The demo has no authentication; bind it to localhost")
    if host == "::1":
        raise ValueError("Use 127.0.0.1 for this IPv4 demo server")
    inference_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/healthz":
                self.send_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [{"id": "gut-rwkv", "object": "model"}]})
            elif self.path == "/":
                content = files("rwkv_jev").joinpath("web/index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self.send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("Request must contain 1–1048576 bytes")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict) or "state" not in request or "questions" not in request:
                    raise ValueError("Request requires state and questions")
                if request.get("model", "gut-rwkv") != "gut-rwkv":
                    raise ValueError("The loaded model is gut-rwkv")
                with inference_lock:
                    result = model.system_one(request["state"], request["questions"])
                self.send_json(200, result)
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as error:
                self.send_json(422, {"error": str(error)})
            except RuntimeError:
                self.send_json(500, {"error": "Model inference failed; check server configuration"})

    return ThreadingHTTPServer((host, port), Handler)


def serve(model, host, port):
    server = create_server(model, host, port)
    print(f"Gut-RWKV demo: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
