"""Mock Acme backend used by the sample tenant's tools (order lookup, ticket creation)."""
import asyncio
import itertools
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

TOKEN = os.environ.get("ACME_API_TOKEN", "acme-dev-token")
DELAY_MS = int(os.environ.get("ACME_LOOKUP_DELAY_MS", "500"))   # simulate a slow legacy system
app = FastAPI(title="Acme mock backend")

ORDERS = {
    "AC48213": {"order_id": "AC48213", "status": "shipped", "carrier": "UPS", "eta": "2026-10-28",
                "items": ["AirBeat Pro earbuds"], "tracking": "1Z999AA10123456784"},
    "AC50007": {"order_id": "AC50007", "status": "processing", "carrier": None, "eta": "2026-11-02",
                "items": ["Nova 14 laptop"], "tracking": None},
    "AC39110": {"order_id": "AC39110", "status": "delivered", "carrier": "FedEx", "eta": "2026-10-20",
                "items": ["Pulse tablet", "Acme Pen"], "tracking": "794644790138"},
}
TICKETS: list[dict] = []
_ids = itertools.count(1001)


def auth(tok: str | None) -> None:
    if tok != TOKEN:
        raise HTTPException(401, "bad token")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/orders/{order_id}")
async def get_order(order_id: str, x_acme_token: str | None = Header(None)):
    auth(x_acme_token)
    await asyncio.sleep(DELAY_MS / 1000)
    o = ORDERS.get(order_id.upper().replace("-", "").replace(" ", ""))
    if not o:
        raise HTTPException(404, "order not found")
    return o


class TicketIn(BaseModel):
    order_id: str | None = None
    subject: str = Field(min_length=3, max_length=200)
    description: str = Field(default="", max_length=2000)
    contact_email: str | None = None


@app.post("/tickets")
async def create_ticket(t: TicketIn, x_acme_token: str | None = Header(None)):
    auth(x_acme_token)
    rec = {"ticket_id": f"T-{next(_ids)}", **t.model_dump(), "status": "open"}
    TICKETS.append(rec)
    return rec


@app.get("/tickets")
async def list_tickets(x_acme_token: str | None = Header(None)):
    auth(x_acme_token)
    return TICKETS
