from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

CAS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
API_BASE = "https://seffaflik.epias.com.tr/electricity-service"

ENDPOINTS = {
    "consumption": "/v1/consumption/data/realtime-consumption",
    "generation": "/v1/generation/data/realtime-generation",
    "mcp": "/v1/markets/dam/data/mcp",
}


class EpiasError(RuntimeError):
    """Raised when EPİAŞ authentication or data retrieval fails."""


@dataclass
class EpiasClient:
    username: str
    password: str
    timeout: float = 45.0
    page_size: int = 1000
    max_pages: int = 100
    transport: httpx.BaseTransport | None = None

    @classmethod
    def from_env(cls) -> "EpiasClient":
        username = os.environ.get("EPIAS_USERNAME")
        password = os.environ.get("EPIAS_PASSWORD")
        if not username or not password:
            raise EpiasError(
                "Set EPIAS_USERNAME and EPIAS_PASSWORD. Credentials are never stored in the repository."
            )
        return cls(username=username, password=password)

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.timeout,
            transport=self.transport,
            follow_redirects=True,
            headers={"User-Agent": "turkiye-grid-foundation-model/0.2-data-audit"},
        )

    def get_tgt(self) -> str:
        with self._client() as client:
            response = client.post(
                CAS_URL,
                data={"username": self.username, "password": self.password},
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "text/plain",
                },
            )
        if response.status_code not in (200, 201):
            raise EpiasError(
                f"EPİAŞ CAS authentication failed: HTTP {response.status_code}. "
                "Check Transparency Platform registration/credentials."
            )
        tgt = response.text.strip().strip('"')
        if not tgt.startswith("TGT-"):
            location = response.headers.get("Location", "")
            if location:
                tgt = location.rstrip("/").split("/")[-1]
        if not tgt.startswith("TGT-"):
            raise EpiasError(
                "EPİAŞ CAS returned HTTP success but no TGT was found in body or Location header."
            )
        return tgt

    @staticmethod
    def _iso(value: str | datetime) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    def fetch(
        self,
        dataset: str,
        start: str | datetime,
        end: str | datetime,
        *,
        tgt: str | None = None,
    ) -> list[dict[str, Any]]:
        if dataset not in ENDPOINTS:
            raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(ENDPOINTS)}")

        tgt = tgt or self.get_tgt()
        endpoint = API_BASE + ENDPOINTS[dataset]
        all_items: list[dict[str, Any]] = []

        with self._client() as client:
            for page_number in range(1, self.max_pages + 1):
                payload = {
                    "startDate": self._iso(start),
                    "endDate": self._iso(end),
                    "page": {
                        "number": page_number,
                        "size": self.page_size,
                        "sort": {"direction": "ASC", "field": "date"},
                    },
                }
                response = client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "TGT": tgt,
                        "Accept": "application/json",
                        "Accept-Language": "en",
                        "Content-Type": "application/json",
                    },
                )
                if response.status_code not in (200, 201):
                    raise EpiasError(
                        f"EPİAŞ {dataset} request failed: HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                body = response.json()
                items = body.get("items", []) if isinstance(body, dict) else []
                if not isinstance(items, list):
                    raise EpiasError(f"EPİAŞ {dataset} response has no list-valued 'items'.")
                all_items.extend(items)
                if len(items) < self.page_size:
                    break
            else:
                raise EpiasError(
                    f"Pagination reached max_pages={self.max_pages}; narrow the requested date range."
                )

        return all_items
