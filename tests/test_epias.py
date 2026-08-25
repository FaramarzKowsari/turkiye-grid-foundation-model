import json

import httpx

from turkiye_grid_fm.epias import API_BASE, CAS_URL, EpiasClient


def test_auth_and_fetch_with_mock_transport():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url) == CAS_URL:
            return httpx.Response(201, text="TGT-123-test")
        if str(request.url) == API_BASE + "/v1/markets/dam/data/mcp":
            body = json.loads(request.content)
            assert body["startDate"].startswith("2025-01-01")
            assert request.headers["TGT"] == "TGT-123-test"
            return httpx.Response(200, json={"items": [{"date": "2025-01-01T00:00:00+03:00", "price": 2000.0}]})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = EpiasClient("u", "p", transport=transport)
    tgt = client.get_tgt()
    items = client.fetch(
        "mcp", "2025-01-01T00:00:00+03:00", "2025-01-01T23:00:00+03:00", tgt=tgt
    )
    assert items[0]["price"] == 2000.0
    assert len(calls) == 2
