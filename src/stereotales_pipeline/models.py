from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from stereotales_pipeline.template import SELF_EVAL_TEMPLATE_VERSION


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


class AssociationEvalJob(BaseModel):
    """Canonical job payload hashed to produce eval_id."""

    template_version: int = SELF_EVAL_TEMPLATE_VERSION
    repetition_index: int
    question_order: Literal["realism_first", "harmful_first"]
    base_attribute: str
    base_value: str
    compared_attribute: str
    compared_value: str


class AssociationEvalResult(BaseModel):
    """Result of one model evaluating one association."""

    eval_id: str
    evaluator_model: str
    job: AssociationEvalJob
    prompt: str
    model_answer: str
    parsed_choice: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
