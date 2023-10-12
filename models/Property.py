from pydantic import BaseModel


class Property(BaseModel):
    """A property implementing pydantic BaseModel"""
    property_type: str
    county: str
    zona: str
    suprafata_utila: float
    suprafata_teren: float
    posibilitate_parcare: str
    nr_locuri_parcare: int
    numar_camere: int
    numar_bai: int
    an_finalizare_constructie: int
