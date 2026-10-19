import urllib.request
import json

def test_endpoints():
    print("[*] Testing /api/status...")
    res = urllib.request.urlopen("http://localhost:8000/api/status")
    status = json.loads(res.read().decode())
    assert status["status"] == "online"
    print("  -> Status OK:", status["system_name"])

    print("[*] Testing /api/code-templates...")
    res = urllib.request.urlopen("http://localhost:8000/api/code-templates")
    templates = json.loads(res.read().decode())["templates"]
    assert "pipeline_demo" in templates
    print("  -> Code templates OK: Found", len(templates), "templates")

    print("[*] Testing /api/run-code sandbox...")
    test_code = "print('Hello from OncoVision AI Code Studio!')\nimport math\nprint('Pi:', round(math.pi, 4))"
    req = urllib.request.Request(
        "http://localhost:8000/api/run-code",
        data=json.dumps({"code": test_code}).encode(),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    out = json.loads(res.read().decode())
    assert out["success"] is True
    print("  -> Sandbox execution OK! Duration:", out["duration_ms"], "ms")
    print("  -> Output:\n" + out["stdout"])

    print("[*] Testing static index.html...")
    res = urllib.request.urlopen("http://localhost:8000/")
    html = res.read().decode()
    assert "OncoVision" in html
    assert "AI Studio" in html
    assert "Interactive Python Code Studio" in html
    print("  -> index.html successfully served! Bytes:", len(html))

    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    test_endpoints()
