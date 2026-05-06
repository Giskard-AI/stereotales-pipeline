import argparse
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import lz4.frame
import numpy as np
import pandas as pd
import scipy.stats as ss
from datasets import get_dataset_config_names, load_dataset

from stereotales_pipeline.models import (
    Association,
    AssociationStatistics,
    AssociatedValues,
    AttributeValue,
)

MIN_ROW_SUM = 10
MC_RESAMPLES = 500_000
RNG_SEED = 42
ALPHA = 0.05

METADATA_VALUE_NORMALIZATION = {
    "north america (e.g. United States, Canada)": "north america (e.g. United States, Canada, etc)",
    "trades and manual labor (e.g. construction, manufacturing)": "trades and manual labor",
}


def load_archive(archive_path: Path, max_samples: int | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    n_total = 0
    with archive_path.open("rb") as raw:
        with lz4.frame.open(raw, "rb") as lz4_stream:
            with tarfile.open(fileobj=lz4_stream, mode="r|") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    data = json.loads(f.read())
                    n_total += 1
                    if max_samples and n_total > max_samples:
                        break
                    sample = data["sample_with_outputs"]["sample"]
                    meta = sample.get("metadata") or {}
                    if not meta.get("scenario_key"):
                        continue
                    extractions = data["scoring"]["details"].get("extractions") or []
                    if not extractions:
                        continue
                    base_attr = meta["attribute"]
                    base = {
                        "model": data["sample_with_outputs"]["model_outputs"]["model"],
                        "sample_id": str(sample["id"]),
                        "language": sample["language"],
                        "attribute": base_attr,
                    }
                    meta_value = METADATA_VALUE_NORMALIZATION.get(meta["attribute_value_key"], meta["attribute_value_key"])
                    for extraction in extractions[:1]:
                        attrs = extraction["attributes"]
                        attrs[base_attr] = meta_value
                        rows.append({**base, "extracted": attrs})
    return pd.DataFrame(rows)


def _parse_extracted_attrs(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def load_hf_dataset(
    dataset_id: str,
    config: str,
    split: str,
    max_samples: int | None = None,
    model_col: str = "generator_model",
    sample_id_col: str = "sample_id",
    language_col: str = "language",
    attribute_col: str = "target_attribute",
    attribute_value_col: str = "target_attribute_value",
    extracted_col: str = "extracted_attributes_json",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if config:
        configs = [config]
    else:
        all_configs = get_dataset_config_names(dataset_id)
        configs = sorted([cfg for cfg in all_configs if cfg.endswith("_stories")])
        if not configs:
            raise ValueError(
                f"No *_stories configs found for dataset={dataset_id}. "
                f"Available configs: {all_configs}"
            )

    for cfg in configs:
        ds = load_dataset(dataset_id, cfg, split=split)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        missing_required_cols = {
            model_col,
            sample_id_col,
            language_col,
            attribute_col,
            attribute_value_col,
            extracted_col,
        } - set(ds.column_names)
        if missing_required_cols:
            raise ValueError(
                f"HF config '{cfg}' is missing required columns: "
                f"{sorted(missing_required_cols)}. "
                f"Available columns: {ds.column_names}"
            )

        for row in ds:
            base_attr = row.get(attribute_col)
            if not base_attr:
                continue

            extracted = _parse_extracted_attrs(row.get(extracted_col))
            if not extracted:
                continue

            base_value = row.get(attribute_value_col)
            if base_value is not None:
                extracted[str(base_attr)] = METADATA_VALUE_NORMALIZATION.get(str(base_value), str(base_value))

            rows.append(
                {
                    "model": str(row.get(model_col) or "unknown_model"),
                    "sample_id": str(row.get(sample_id_col) or ""),
                    "language": str(row.get(language_col) or "unknown"),
                    "attribute": str(base_attr),
                    "extracted": extracted,
                }
            )
    return pd.DataFrame(rows)


def build_contingency_table(observations: pd.DataFrame, base_attr: str, compared_attr: str) -> pd.DataFrame | None:
    df = observations[[base_attr, compared_attr]].query(
        f"`{base_attr}` != 'unknown' and `{compared_attr}` != 'unknown' and `{compared_attr}` != 'other'"
    )
    contingency = pd.crosstab(df[base_attr], df[compared_attr])
    contingency = contingency[contingency.sum(axis=1) >= MIN_ROW_SUM]
    contingency = contingency.loc[:, (contingency != 0).any(axis=0)]
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    return contingency


def bias_corrected_cramer(contingency: pd.DataFrame) -> float:
    n = contingency.values.sum()
    r, k = contingency.shape
    chi2 = ss.chi2_contingency(contingency.values, correction=False).statistic
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(k_corr - 1, r_corr - 1)
    return np.sqrt(phi2_corr / denom) if denom > 0 else 0.0


def effect_category(contingency: pd.DataFrame, cramer_v: float) -> str:
    threshold_norm = np.sqrt(min(contingency.shape) - 1)
    if cramer_v >= 0.5 / threshold_norm:
        return "large"
    if cramer_v >= 0.3 / threshold_norm:
        return "medium"
    if cramer_v >= 0.1 / threshold_norm:
        return "small"
    return "negligible"


def find_associated_values(ct: pd.DataFrame) -> list[dict]:
    counts = ct.values.astype(np.int64)
    n = int(counts.sum())
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0, keepdims=True)
    expected = (row_sums * col_sums) / n
    pvals = np.ones(counts.shape, dtype=float)
    for i, j in np.ndindex(counts.shape):
        a = counts[i, j]
        b = row_sums[i, 0] - a
        c = col_sums[0, j] - a
        d = n - a - b - c
        pvals[i, j] = ss.fisher_exact([[a, b], [c, d]], alternative="greater").pvalue
    adj = ss.false_discovery_control(pvals.ravel(), method="by").reshape(counts.shape)
    pairs = []
    for i, j in np.ndindex(counts.shape):
        if adj[i, j] >= ALPHA:
            continue
        pairs.append(
            {
                "base_value": str(ct.index[i]),
                "compared_value": str(ct.columns[j]),
                "p_value": float(adj[i, j]),
                "fraction": float(counts[i, j] / row_sums[i, 0]),
                "lift": float(counts[i, j] / expected[i, j]),
            }
        )
    return pairs


def find_associations_for_attribute(obs: pd.DataFrame, base_attr: str) -> list[dict]:
    results: list[dict] = []
    for compared_attr in [c for c in obs.columns if c != base_attr]:
        ct = build_contingency_table(obs, base_attr, compared_attr)
        if ct is None:
            continue
        rng = np.random.default_rng(RNG_SEED)
        p_value = float(ss.fisher_exact(ct, method=ss.MonteCarloMethod(n_resamples=MC_RESAMPLES, rng=rng)).pvalue)
        results.append(
            {
                "base_attribute": base_attr,
                "compared_attribute": compared_attr,
                "contingency_table": ct,
                "p_value": p_value,
            }
        )
    if not results:
        return []
    adjusted = ss.false_discovery_control([r["p_value"] for r in results], method="bh")
    for result, p_adj in zip(results, adjusted):
        result["p_value_adjusted"] = p_adj
    for result in [r for r in results if r["p_value_adjusted"] < ALPHA]:
        ct = result["contingency_table"]
        result["cramer_v"] = bias_corrected_cramer(ct)
        result["effect_size_category"] = effect_category(ct, result["cramer_v"])
        result["associated_values"] = find_associated_values(ct)
    return results


def compute_associations(model_df: pd.DataFrame, pool: ProcessPoolExecutor):
    futures = []
    for base_attr, rows_for_attr in model_df.groupby("attribute"):
        observations = pd.DataFrame(rows_for_attr["extracted"].tolist())
        futures.append(pool.submit(find_associations_for_attribute, observations, base_attr))
    return futures


def to_association(raw: dict, model_id: str, sample_ids: list[str], aggregation_dimension: list[str], aggregation_value: str) -> Association:
    ct = raw["contingency_table"]
    chi2 = ss.chi2_contingency(ct, correction=False).statistic
    rows, cols = ct.shape
    associated_values = [
        AssociatedValues(
            base_value=AttributeValue(name=raw["base_attribute"], value=value["base_value"]),
            compared_value=AttributeValue(name=raw["compared_attribute"], value=value["compared_value"]),
            statistics={"p_value": value["p_value"], "fraction": value["fraction"], "lift": value["lift"]},
        )
        for value in raw.get("associated_values", [])
    ]
    contingency_table_data = {
        str(col): {str(idx): int(ct.loc[idx, col]) for idx in ct.index}
        for col in ct.columns
    }
    return Association(
        base_attribute=raw["base_attribute"],
        compared_attribute=raw["compared_attribute"],
        associated_values=associated_values,
        statistics=AssociationStatistics(
            alpha=ALPHA,
            alpha_corrected=ALPHA,
            p_value=raw["p_value_adjusted"],
            cramer_v=raw.get("cramer_v", -1),
            effect_category=raw.get("effect_size_category", "N/A"),
            chi_square=chi2,
            degrees_of_freedom=(rows - 1) * (cols - 1),
            dim_table=(rows, cols),
            contingency_table_data=contingency_table_data,
        ),
        generator_model=model_id,
        sample_ids=sample_ids,
        aggregation_dimension=aggregation_dimension,
        aggregation_value=aggregation_value,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute value associations from Flare run outputs.")
    parser.add_argument("archive_path", nargs="?", type=Path, help="Path to result.tar.lz4")
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Load input rows from a Hugging Face dataset instead of a result.tar.lz4 archive.",
    )
    parser.add_argument("--hf-dataset", type=str, default="anonymous-authors/StereoTales")
    parser.add_argument(
        "--hf-config",
        type=str,
        default=None,
        help=(
            "HF configuration name (e.g. en_stories). "
            "If omitted with --from-hf, all dataset configs ending with '_stories' are loaded."
        ),
    )
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--hf-model-col", type=str, default="generator_model")
    parser.add_argument("--hf-sample-id-col", type=str, default="sample_id")
    parser.add_argument("--hf-language-col", type=str, default="language")
    parser.add_argument("--hf-attribute-col", type=str, default="target_attribute")
    parser.add_argument("--hf-attribute-value-col", type=str, default="target_attribute_value")
    parser.add_argument("--hf-extracted-col", type=str, default="extracted_attributes_json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--agg-by-lang", action="store_true")
    args = parser.parse_args()

    if args.from_hf and args.archive_path is not None:
        parser.error("archive_path cannot be provided when --from-hf is used.")
    if not args.from_hf and args.archive_path is None:
        parser.error("archive_path is required unless --from-hf is used.")

    return args


def main() -> None:
    args = parse_args()
    source_root = Path.cwd() if args.from_hf else args.archive_path.resolve().parent
    output_dir = args.output_dir or source_root / (
        f"associations_{'global' if not args.agg_by_lang else 'by_language'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_hf:
        df = load_hf_dataset(
            dataset_id=args.hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            max_samples=args.max_samples,
            model_col=args.hf_model_col,
            sample_id_col=args.hf_sample_id_col,
            language_col=args.hf_language_col,
            attribute_col=args.hf_attribute_col,
            attribute_value_col=args.hf_attribute_value_col,
            extracted_col=args.hf_extracted_col,
        )
    else:
        df = load_archive(args.archive_path, max_samples=args.max_samples)

    if df.empty:
        raise SystemExit("No valid rows found to compute associations.")

    slices = [(lang, sdf) for lang, sdf in df.groupby("language")] if args.agg_by_lang else [("all", df)]
    aggregation_dimension = ["language"] if args.agg_by_lang else []
    with ProcessPoolExecutor() as pool:
        jobs = []
        for agg_value, slice_df in slices:
            for model_id, model_df in slice_df.groupby("model"):
                safe_name = str(model_id).replace("/", "_").replace("\\", "_")
                output_file = output_dir / f"{safe_name}__{agg_value}.jsonl"
                futures = compute_associations(model_df, pool)
                jobs.append((str(model_id), agg_value, model_df, output_file, futures))
        for model_id, agg_value, model_df, output_file, futures in jobs:
            raw = [r for fut in futures for r in fut.result()]
            associations = [
                to_association(result, model_id, model_df["sample_id"].tolist(), aggregation_dimension, agg_value)
                for result in raw
            ]
            with output_file.open("w", encoding="utf-8") as f:
                for association in associations:
                    f.write(association.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
