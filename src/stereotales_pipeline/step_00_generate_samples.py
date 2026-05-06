import argparse
import itertools
import uuid
from pathlib import Path
from typing import Any

import yaml
from flare.schema import EvaluationData, Generation, GenerationParams, Sample

TEMPERATURE = 1.0
NUM_REPEATS_SCENARIO = 1
NUM_REPEATS_PROMPT = 15

BASE_ATTRIBUTES = [
    "age",
    "gender",
    "gender_alignment",
    "sexual_orientation",
    "geographic_origin",
    "religion",
    "political_orientation",
    "disability_status",
    "immigration_status",
    "urbanicity",
    "marital_status",
    "parental_status",
    "income_level",
    "employment_status",
    "professional_field",
    "education",
    "health_status",
    "literacy_status",
    "housing_status",
]


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _create_generation(
    prompt_template: str,
    prompt_template_key: str,
    character: str,
    attribute: str,
    attribute_value: str,
    language: str,
    scenario: dict[str, str] | None = None,
) -> Generation:
    if scenario:
        prompt = prompt_template.format(context=scenario["scenario"], character=character)
    else:
        prompt = prompt_template.format(character=character)

    return Generation(
        id=uuid.uuid4(),
        type="chat_completion",
        messages=[{"role": "user", "content": prompt}],
        params=GenerationParams(temperature=TEMPERATURE, max_tokens=4096),
        metadata={
            "language": language,
            "attribute": attribute,
            "attribute_value": attribute_value,
            "scenario": scenario["scenario"] if scenario else None,
            "scenario_group": scenario["group"] if scenario else None,
            "character": character,
            "prompt_template": prompt_template_key,
        },
    )


def _sample(
    language: str,
    attribute: str,
    attribute_value: str,
    attribute_value_key: str,
    character: str,
    prompt_template_key: str,
    generations: list[Generation],
    scenario_key: str | None = None,
    scenario_group: str | None = None,
) -> Sample:
    return Sample(
        id=uuid.uuid4(),
        module="biases",
        task="story_generation",
        language=language,
        generations=generations,
        metadata={
            "attribute": attribute,
            "attribute_value": attribute_value,
            "attribute_value_key": attribute_value_key,
            "scenario_key": scenario_key,
            "scenario_group": scenario_group,
            "character": character,
            "temperature": TEMPERATURE,
            "num_repeats": len(generations),
            "prompt_template": prompt_template_key,
        },
        evaluation=EvaluationData(
            scorer="biases/attribute_extraction",
            data={"attribute": attribute, "language": language},
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Flare-compatible story samples.")
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attrs_map: dict[str, dict[str, Any]] = {}
    prompts_map: dict[str, dict[str, str]] = {}
    scenarios_map: dict[str, list[dict[str, str]]] = {}

    for lang in set(["en", *args.languages]):
        attrs_map[lang] = _load_yaml(args.seed_dir / "attributes" / f"attributes.{lang}.yaml")
        prompts_map[lang] = _load_yaml(args.seed_dir / "prompts_template" / f"prompts.{lang}.yaml")
        raw_scenarios = _load_yaml(args.seed_dir / "scenarios" / f"scenario.{lang}.yaml")
        scenarios_map[lang] = [
            {"scenario": scenario, "group": group}
            for group, scenarios in raw_scenarios.items()
            for scenario in (scenarios or [])
        ]

    additional_prompts = _load_yaml(args.seed_dir / "prompts_template" / "additional_prompt.en.yaml")
    en_attrs = attrs_map["en"]
    en_scenarios = scenarios_map["en"]

    for lang in args.languages:
        samples: list[Sample] = []
        for attribute in BASE_ATTRIBUTES:
            attr_data = attrs_map[lang].get(attribute)
            if not attr_data:
                continue

            values = attr_data.get("values", [])
            template_choices = attr_data.get("template_choices", [])
            en_values = en_attrs.get(attribute, {}).get("values", [])
            valid_choices = [
                (value, choice, en_values[i] if i < len(en_values) else value)
                for i, (value, choice) in enumerate(zip(values, template_choices))
                if choice is not None
            ]
            for idx, scenario in enumerate(scenarios_map[lang]):
                scenario_key = en_scenarios[idx]["scenario"] if idx < len(en_scenarios) else scenario["scenario"]
                for value, character, value_key in valid_choices:
                    generations = [
                        _create_generation(
                            prompt_template=prompts_map[lang]["scenario_based"],
                            prompt_template_key="scenario_based",
                            scenario=scenario,
                            character=character,
                            attribute=attribute,
                            attribute_value=value,
                            language=lang,
                        )
                        for _ in range(NUM_REPEATS_SCENARIO)
                    ]
                    samples.append(
                        _sample(
                            language=lang,
                            attribute=attribute,
                            attribute_value=value,
                            attribute_value_key=value_key,
                            character=character,
                            prompt_template_key="scenario_based",
                            generations=generations,
                            scenario_key=scenario_key,
                            scenario_group=scenario["group"],
                        )
                    )

        out = args.output_dir / f"story_generation_samples.{lang}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample.model_dump_json() + "\n")

    if "en" in args.languages:
        additional_samples: list[Sample] = []
        valid_templates = {k: v for k, v in additional_prompts.items() if v}
        for attribute in BASE_ATTRIBUTES:
            attr_data = en_attrs.get(attribute)
            if not attr_data:
                continue
            valid_choices = [
                (value, choice)
                for value, choice in zip(attr_data.get("values", []), attr_data.get("template_choices", []))
                if choice is not None
            ]
            for (value, character), (template_key, template) in itertools.product(valid_choices, valid_templates.items()):
                generations = [
                    _create_generation(
                        prompt_template=template,
                        prompt_template_key=template_key,
                        character=character,
                        attribute=attribute,
                        attribute_value=value,
                        language="en",
                    )
                    for _ in range(NUM_REPEATS_PROMPT)
                ]
                additional_samples.append(
                    _sample(
                        language="en",
                        attribute=attribute,
                        attribute_value=value,
                        attribute_value_key=value,
                        character=character,
                        prompt_template_key=template_key,
                        generations=generations,
                    )
                )
        out = args.output_dir / "story_generation_samples.additional_en.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for sample in additional_samples:
                f.write(sample.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
