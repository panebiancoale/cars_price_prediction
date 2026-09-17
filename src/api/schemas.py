from typing import List
from pydantic import BaseModel


class CarsFeatures(BaseModel):
    Brand: str
    Model: str
    Country: str
    Kilometers: float
    Gearbox: str
    Year: int
    Fuel: str
    Power: float
    Seller: str
    Drivetrain: str
    Color: str

class PredictionRequest(BaseModel):
    data: List[CarsFeatures]
