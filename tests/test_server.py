import json
from threading import Thread
import urllib.error
import urllib.request

import pytest

from rwkv_jev.server import create_server


class DemoModel:
    def system_one(self, state, questions):
        return {"model": "gut-rwkv", "answers": {name: {"type": "noul", "noul": 0.75} for name in questions},
                "generated_tokens": 0}


def test_local_server_serves_demo_and_typed_requests():
    server = create_server(DemoModel(), port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(address) as response:
            assert b"Gut-RWKV" in response.read()
        request = {"model": "gut-rwkv", "state": "yes", "questions": {"decision": {"type": "noul", "instructions": "yes?"}}}
        with urllib.request.urlopen(address + "/v1/systemone", data=json.dumps(request).encode()) as response:
            assert json.load(response)["answers"]["decision"]["noul"] == 0.75
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(address + "/v1/systemone", data=b"{}")
        assert error.value.code == 422
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(address + "/v1/systemone", data=b"not-json")
        assert error.value.code == 422
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_server_rejects_nonlocal_binding():
    with pytest.raises(ValueError, match="localhost"):
        create_server(DemoModel(), host="0.0.0.0")
