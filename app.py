"""
Georgian Real Estate Voice Agent — tools backend for Vapi.

Speaks two protocols on the same endpoints:
  1. Vapi custom-tool format:  {"message": {"toolCallList": [{id, name, arguments}]}}
     -> replies {"results": [{"toolCallId": ..., "result": "<georgian text>"}]}
  2. Plain JSON body (for curl / local testing / /docs)

Anti-fabrication is the core design goal: the model only ever receives
verified rows from inventory.csv, rendered as Georgian text. When nothing
matches, it receives an explicit Georgian instruction NOT to invent options.
"""

import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="Georgian Real Estate Voice Agent Tools", version="2.0.0")

BASE = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.environ.get("INVENTORY_PATH", os.path.join(BASE, "inventory.csv"))
LEADS = os.environ.get("LEADS_PATH", os.path.join(BASE, "leads.jsonl"))
# Shared secret. If unset, auth is disabled (local dev only).
API_SECRET = os.environ.get("API_SECRET", "").strip()
MAX_RESULTS = 3  # keep it short — this is a phone call, not a web page

# --------------------------------------------------------------------------
# Georgian term normalisation
# The LLM is told to send English enum values, but callers speak Georgian and
# models leak source-language terms. Map both so a filter never silently fails.
# --------------------------------------------------------------------------

VIEW_MAP = {
    "sea": "sea", "ზღვ": "sea", "ზღვის": "sea", "ზღვა": "sea",
    "city": "city", "ქალაქ": "city", "ქალაქის": "city",
    "mountain": "mountain", "მთ": "mountain", "მთის": "mountain",
    "park": "park", "პარკ": "park",
    "yard": "yard", "ეზო": "yard",
}

VIEW_KA = {
    "sea": "ზღვის ხედი",
    "city": "ქალაქის ხედი",
    "mountain": "მთის ხედი",
    "park": "პარკის ხედი",
    "yard": "ეზოს ხედი",
}

PAYMENT_KA = {
    "full": "სრული გადახდა",
    "installment": "შიდა განვადება",
    "mortgage": "იპოთეკა",
}


def normalise_view(value: Any) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    for key, canonical in VIEW_MAP.items():
        if key in v:
            return canonical
    return v


def to_float(value: Any) -> Optional[float]:
    """Tolerant number parsing — models send '80 000', '$80,000', '80k'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower().replace(",", "").replace(" ", "").replace("$", "")
    s = s.replace("aშ", "").replace("usd", "").replace("დოლარი", "")
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1000.0, s[:-1]
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) * mult if m else None


def to_int(value: Any) -> Optional[int]:
    f = to_float(value)
    return int(f) if f is not None else None


def fmt_money(value: float) -> str:
    """80000 -> '80 000' — readable when spoken aloud."""
    return f"{int(round(value)):,}".replace(",", " ")


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "project", "unit_id", "status", "floor", "area_m2",
    "bedrooms", "view", "price_usd",
}


def load_inventory() -> list[dict]:
    if not os.path.exists(INVENTORY):
        return []
    with open(INVENTORY, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def inventory_health() -> dict:
    rows = load_inventory()
    if not rows:
        return {"rows": 0, "available": 0, "columns_ok": False, "demo_data": False}
    cols = set(rows[0].keys())
    demo = any("demo" in (r.get("project", "") + r.get("notes", "")).lower() for r in rows)
    return {
        "rows": len(rows),
        "available": sum(1 for r in rows if r.get("status", "").lower() == "available"),
        "columns_ok": REQUIRED_COLUMNS.issubset(cols),
        "missing_columns": sorted(REQUIRED_COLUMNS - cols),
        "demo_data": demo,
    }


def render_unit_ka(r: dict) -> str:
    """One apartment as a short Georgian sentence a salesperson would say."""
    price = to_float(r.get("price_usd"))
    area = to_float(r.get("area_m2"))
    beds = to_int(r.get("bedrooms"))
    view = normalise_view(r.get("view"))

    parts = [f"{r.get('project', '')} — ბინა {r.get('unit_id', '')}"]
    if beds is not None:
        parts.append(f"{beds} საძინებელი")
    if area is not None:
        parts.append(f"{area:g} კვ.მ")
    if r.get("floor"):
        parts.append(f"{r['floor']}-ე სართული")
    if view:
        parts.append(VIEW_KA.get(view, view))
    if price is not None:
        parts.append(f"{fmt_money(price)} დოლარი")

    line = ", ".join(parts)

    dp = to_float(r.get("down_payment_percent"))
    months = to_int(r.get("installment_months"))
    if dp is not None and months:
        line += f". განვადება: {dp:g}% პირველადი შენატანი, {months} თვე"
    return line


def matches(r: dict, q: dict) -> bool:
    if r.get("status", "").lower() != "available":
        return False
    if q.get("project"):
        if str(q["project"]).lower() not in r.get("project", "").lower():
            return False
    if q.get("max_price_usd") is not None:
        price = to_float(r.get("price_usd"))
        if price is None or price > q["max_price_usd"]:
            return False
    if q.get("min_area_m2") is not None:
        area = to_float(r.get("area_m2"))
        if area is None or area < q["min_area_m2"]:
            return False
    if q.get("bedrooms") is not None:
        beds = to_int(r.get("bedrooms"))
        if beds is None or beds != q["bedrooms"]:
            return False
    if q.get("view"):
        if normalise_view(r.get("view")) != q["view"]:
            return False
    return True


def do_search(raw: dict) -> dict:
    q = {
        "project": (raw.get("project") or None),
        "max_price_usd": to_float(raw.get("max_price_usd")),
        "min_area_m2": to_float(raw.get("min_area_m2")),
        "bedrooms": to_int(raw.get("bedrooms")),
        "view": normalise_view(raw.get("view")),
    }

    rows = load_inventory()
    if not rows:
        return {
            "count": 0,
            "results": [],
            "speak_ka": "ბაზა ამჟამად მიუწვდომელია. უთხარი კლიენტს, რომ დეტალებს "
                        "გადაამოწმებ და მენეჯერი დაუკავშირდება. არ დაასახელო "
                        "არცერთი კონკრეტული ბინა, ფასი ან პირობა.",
        }

    hits = [r for r in rows if matches(r, q)]
    hits.sort(key=lambda x: to_float(x.get("price_usd")) or 0.0)
    top = hits[:MAX_RESULTS]

    if top:
        lines = [render_unit_ka(r) for r in top]
        speak = (
            f"ნაპოვნია {len(hits)} შესაბამისი ვარიანტი. "
            f"დაასახელე მაქსიმუმ ორი, მოკლედ, შემდეგი სიიდან — "
            f"არ შეცვალო ციფრები და არ დაამატო არარსებული დეტალი:\n"
            + "\n".join(f"- {ln}" for ln in lines)
        )
        return {
            "count": len(hits),
            "results": top,
            "speak_ka": speak,
        }

    # No exact match. Offer clearly-labelled near matches — never as exact hits.
    relaxed = {**q, "view": None, "bedrooms": None}
    near = [r for r in rows if matches(r, relaxed)]
    near.sort(key=lambda x: to_float(x.get("price_usd")) or 0.0)
    near = near[:2]

    if near:
        lines = [render_unit_ka(r) for r in near]
        speak = (
            "ზუსტად ამ პარამეტრებით თავისუფალი ბინა არ არის. "
            "აუცილებლად უთხარი კლიენტს, რომ ზუსტი დამთხვევა არ მოიძებნა, "
            "და მხოლოდ ამის შემდეგ შესთავაზე ეს ახლოს მდგომი ვარიანტები:\n"
            + "\n".join(f"- {ln}" for ln in lines)
        )
    else:
        speak = (
            "ამ პარამეტრებით თავისუფალი ბინა არ მოიძებნა. "
            "უთხარი კლიენტს პირდაპირ, რომ ამჟამად შესაბამისი ვარიანტი არ არის, "
            "და შესთავაზე ბიუჯეტის ან პარამეტრების შეცვლა. "
            "არ მოიგონო ბინა, ფასი ან ხელმისაწვდომობა."
        )

    return {"count": 0, "results": [], "near_matches": near, "speak_ka": speak}


# --------------------------------------------------------------------------
# Leads
# --------------------------------------------------------------------------

def normalise_phone(value: Any) -> Optional[str]:
    if not value:
        return None
    digits = re.sub(r"[^\d+]", "", str(value))
    if digits.startswith("995") and len(digits) == 12:
        return "+" + digits
    if len(digits) == 9 and digits.startswith("5"):
        return "+995" + digits
    return digits or None


VALID_TEMPS = {"hot", "warm", "cold"}


def do_lead(raw: dict) -> dict:
    temp = str(raw.get("lead_temperature") or "").strip().lower()
    item = {
        "name": raw.get("name") or None,
        "phone": normalise_phone(raw.get("phone")),
        "project": raw.get("project") or None,
        "budget_usd": to_float(raw.get("budget_usd")),
        "desired_area_m2": to_float(raw.get("desired_area_m2")),
        "bedrooms": to_int(raw.get("bedrooms")),
        "view": normalise_view(raw.get("view")),
        "timeline": raw.get("timeline") or None,
        "payment_type": raw.get("payment_type") or None,
        "interest_unit_id": raw.get("interest_unit_id") or None,
        "notes": raw.get("notes") or None,
        "lead_temperature": temp.capitalize() if temp in VALID_TEMPS else "Warm",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(LEADS, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except OSError as exc:
        return {
            "saved": False,
            "error": str(exc),
            "speak_ka": "ლიდი ვერ შევინახე. უთხარი კლიენტს, რომ მენეჯერი "
                        "მალე დაუკავშირდება, და გადაამოწმე ტელეფონის ნომერი.",
        }

    missing = [k for k in ("name", "phone") if not item[k]]
    if missing:
        speak = ("ლიდი შენახულია, მაგრამ აკლია: "
                 + ", ".join("სახელი" if m == "name" else "ტელეფონი" for m in missing)
                 + ". თავაზიანად დააზუსტე ეს ინფორმაცია.")
    else:
        speak = (f"ლიდი შენახულია. დაუდასტურე კლიენტს, რომ მენეჯერი "
                 f"დაუკავშირდება ნომერზე {item['phone']}.")

    return {"saved": True, "lead": item, "speak_ka": speak}


# --------------------------------------------------------------------------
# Vapi protocol adapter
# --------------------------------------------------------------------------

def extract_tool_calls(body: dict) -> list[tuple[str, str, dict]]:
    """Return [(toolCallId, name, arguments)] for both Vapi payload shapes."""
    msg = (body or {}).get("message") or {}
    calls = []

    for tc in msg.get("toolCallList") or []:
        args = tc.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append((tc.get("id", ""), tc.get("name", ""), args))

    if not calls:  # older OpenAI-style shape
        for tc in msg.get("toolCalls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append((tc.get("id", ""), fn.get("name", ""), args))

    return calls


def check_auth(secret: Optional[str]) -> None:
    if API_SECRET and (secret or "").strip() != API_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")


async def handle(request: Request, handler, x_vapi_secret: Optional[str]) -> Any:
    check_auth(x_vapi_secret)
    try:
        body = await request.json()
    except Exception:
        body = {}

    calls = extract_tool_calls(body)
    if calls:
        results = []
        for call_id, _name, args in calls:
            out = handler(args)
            results.append({"toolCallId": call_id, "result": out["speak_ka"]})
        return {"results": results}

    # Plain JSON — local testing and /docs
    return handler(body if isinstance(body, dict) else {})


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

class SearchRequest(BaseModel):
    project: Optional[str] = None
    max_price_usd: Optional[float] = None
    min_area_m2: Optional[float] = None
    bedrooms: Optional[int] = None
    view: Optional[str] = None


class LeadRequest(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    project: Optional[str] = None
    budget_usd: Optional[float] = None
    desired_area_m2: Optional[float] = None
    bedrooms: Optional[int] = None
    view: Optional[str] = None
    timeline: Optional[str] = None
    payment_type: Optional[str] = None
    interest_unit_id: Optional[str] = None
    notes: Optional[str] = None
    lead_temperature: Optional[str] = None


@app.get("/health")
def health():
    inv = inventory_health()
    return {
        "ok": True,
        "version": app.version,
        "auth_enabled": bool(API_SECRET),
        "inventory": inv,
        "warning": (
            "DEMO INVENTORY IS LOADED — replace inventory.csv before taking real calls."
            if inv.get("demo_data") else None
        ),
    }


@app.post("/search-apartments")
async def search_apartments(request: Request, x_vapi_secret: Optional[str] = Header(None)):
    return await handle(request, do_search, x_vapi_secret)


@app.post("/create-lead")
async def create_lead(request: Request, x_vapi_secret: Optional[str] = Header(None)):
    return await handle(request, do_lead, x_vapi_secret)


@app.get("/leads")
def list_leads(x_vapi_secret: Optional[str] = Header(None)):
    check_auth(x_vapi_secret)
    if not os.path.exists(LEADS):
        return {"count": 0, "leads": []}
    with open(LEADS, encoding="utf-8") as f:
        leads = [json.loads(line) for line in f if line.strip()]
    return {"count": len(leads), "leads": leads}
