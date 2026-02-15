"""
Pydantic models for request and response validation
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Annotated
from enum import Enum


class HeatingType(str, Enum):
    """Heating types available in the dataset"""
    ELEKTRYCZNE = "elektryczne"
    GAZOWE = "gazowe"
    INNE = "inne"
    KOMINKOWE_GAZOWE = "kominkowe\ngazowe"
    KOTLOWNIA = "kotłownia"
    MIEJSKIE = "miejskie"
    POMPA_CIEPLA = "pompa ciepła"
    WEGLOWE = "węglowe"


class BuildingMaterialType(str, Enum):
    """Building material types available in the dataset"""
    BETON = "beton"
    BETON_KOMOROWY = "beton komórkowy"
    CEGLA = "cegła"
    DREWNO = "drewno"
    INNY = "inny"
    KERAMZYT = "keramzyt"
    PUSTAK = "pustak"
    SILIKAT = "silikat"
    WIELKA_PLYTA = "wielka płyta"
    ZELBET = "żelbet"


class BuildingType(str, Enum):
    """Building types available in the dataset"""
    APARTAMENTOWIEC = "apartamentowiec"
    BLIZNIA = "bliźniak"
    BLOK = "blok"
    DOM_WOLNOSTOJACY = "dom wolnostojący"
    KAMIENICA = "kamienica"
    LOFT = "loft"
    PLOMBA = "plomba"
    SZEREGOWIEC = "szeregowiec"
    WOLNOSTOJACY = "wolnostojący"


class MarketType(str, Enum):
    """Market types available in the dataset"""
    PIERWOTNY = "pierwotny"
    WTORNY = "wtórny"


class Voivodeship(str, Enum):
    """Polish voivodeships (provinces)"""
    DOLNOSLASKIE = "dolnośląskie"
    KUJAWSKO_POMORSKIE = "kujawsko-pomorskie"
    LUBELSKIE = "lubelskie"
    LUBUSKIE = "lubuskie"
    MAZOWIECKIE = "mazowieckie"
    MALOPOLSKIE = "małopolskie"
    OPOLSKIE = "opolskie"
    PODKARPACKIE = "podkarpackie"
    PODLASKIE = "podlaskie"
    POMORSKIE = "pomorskie"
    WARMINSKO_MAZURSKIE = "warmińsko-mazurskie"
    WIELKOPOLSKIE = "wielkopolskie"
    ZACHODNIOPOMORSKIE = "zachodniopomorskie"
    LODZKIE = "łódzkie"
    SLASKIE = "śląskie"
    SWIETOKRZYSKIE = "świętokrzyskie"


class PredictionRequest(BaseModel):
    """Request model for price prediction"""
    area: float = Field(..., gt=0, description="Property area in m²", example=120.5)
    rooms: int = Field(..., gt=0, description="Number of rooms", example=4)
    year_constructed: int = Field(..., ge=1900, le=2025, description="Year when property was built", example=2020)
    heating: HeatingType = Field(..., description="Type of heating system", example="gazowe")
    building_material: BuildingMaterialType = Field(..., description="Building material", example="cegła")
    building_type: BuildingType = Field(..., description="Type of building", example="bliźniak")
    market: MarketType = Field(..., description="Primary or secondary market", example="pierwotny")
    voivodeship: Voivodeship = Field(..., description="Voivodeship (province)", example="mazowieckie")
    city: Annotated[Optional[str], Field(None, description="City name (optional, for local statistics)", example="Kraków")] = None
    district: Annotated[Optional[str], Field(None, description="District name (optional, for local statistics)", example="Wawer")] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "area": 120.5,
                "rooms": 4,
                "year_constructed": 2020,
                "heating": "gazowe",
                "building_material": "cegła",
                "building_type": "bliźniak",
                "market": "pierwotny",
                "voivodeship": "mazowieckie",
                "city": "Kraków",
                "district": "Wawer"
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response model for price prediction"""
    predicted_price: float = Field(..., description="Predicted property price in PLN")
    currency: str = Field(default="PLN", description="Currency")
    confidence: str = Field(..., description="Model confidence level")
    input_features: PredictionRequest
    local_stats: Optional[dict] = Field(None, description="Local statistics for the city/district if provided")


class FilterRequest(BaseModel):
    """Request to filter available properties by location"""
    voivodeship: Optional[Voivodeship] = Field(None, description="Filter by voivodeship")
    city: Optional[str] = Field(None, description="Filter by city name")
    district: Optional[str] = Field(None, description="Filter by district name")


class ModelInfo(BaseModel):
    """Information about the trained model"""
    model_type: str
    algorithm: str
    features_used: List[str]
    training_samples: int
    test_r2_score: float
    test_rmse: float
    test_mae: float
    last_updated: str
