from fastapi import APIRouter
from app.models.barcode import BarcodeTeachReq
from app.services.common import (
    load_profiles,
    canonical_from_name,
    set_barcode_mapping,
    get_barcode_mapping,
)


router = APIRouter(prefix="/barcode")

@router.post("/teach")
async def barcode_teach(req: BarcodeTeachReq):
    profiles = load_profiles()
    canon = canonical_from_name(req.label, profiles)
    set_barcode_mapping(req.barcode.strip(), req.label.strip(), canon)
    return {"ok": True, "barcode": req.barcode.strip(), "label": req.label.strip(), "canonical": canon}

# NOTE: This endpoint works best for numeric barcodes.
# URLs contain slashes and may not work nicely in a path parameter.
# Use /barcode/teach for URLs (it stores fine), and verify via DB or scans.
@router.get("/resolve/{barcode}")
async def barcode_resolve(barcode: str):
    m = get_barcode_mapping(barcode.strip())
    if not m:
        return {"ok": False, "known": False, "barcode": barcode.strip()}
    return {"ok": True, "known": True, "barcode": m["barcode"], "label": m["label"], "canonical": m["canonical"]}

