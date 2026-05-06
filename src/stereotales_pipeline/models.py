from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class AttributeValue(BaseModel):
    name: str
    value: str


class AssociatedValues(BaseModel):
    base_value: AttributeValue
    compared_value: AttributeValue
    statistics: dict[str, Any]


class AssociationStatistics(BaseModel):
    model_config = ConfigDict(extra="allow")

    alpha: float
    alpha_corrected: float | None = None
    p_value: float
    cramer_v: float
    effect_category: str
    chi_square: float
    degrees_of_freedom: int
    dim_table: tuple[int, int]
    contingency_table_data: dict[str, dict[str, int]]


class Association(BaseModel):
    base_attribute: str
    compared_attribute: str
    associated_values: list[AssociatedValues]
    statistics: AssociationStatistics
    generator_model: str
    sample_ids: list[str]
    aggregation_dimension: list[Literal["scenario", "language", "attribute"]]
    aggregation_value: Any
