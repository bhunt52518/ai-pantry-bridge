from pydantic import BaseModel

class BarcodeTeachReq(BaseModel):
    barcode: str
    label: str
