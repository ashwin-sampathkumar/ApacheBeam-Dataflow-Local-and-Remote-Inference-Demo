# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Production Apache Beam Bulk ML Inference Pipeline.

Combines built-in remote Gemini ModelHandlers (for structured text & image generation)
with local GPU safety moderation using ShieldGemma 2 on Google Cloud Dataflow.
"""

import argparse
import datetime
import gc
import inspect
import io
import json
import logging
import os
import re
import sys
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import (
    KeyedModelHandler,
    ModelHandler,
    PredictionResult,
    RunInference,
)
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.utils import multi_process_shared, shared as beam_shared
import numpy as np
from PIL import Image, ImageDraw

# Built-in Apache Beam Gemini ModelHandlers
from apache_beam.ml.inference.gemini_inference import (
    GeminiModelHandler,
    generate_from_string,
)
try:
    from apache_beam.ml.inference.gemini_inference import (
        generate_image_from_strings_and_images,
    )
except ImportError:
    def generate_image_from_strings_and_images(
        model_name: str, batch: Sequence[Any], model: Any, inference_args: Dict[str, Any]
    ):
        return model.models.generate_content(
            model=model_name, contents=batch, **inference_args
        )

# Optional PyTorch & Transformers dependencies for local ShieldGemma 2 GPU inference
try:
    import torch
    from transformers import AutoProcessor, ShieldGemma2ForImageClassification
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        BitsAndBytesConfig = None
except ImportError:
    torch = None
    AutoProcessor = None
    ShieldGemma2ForImageClassification = None
    BitsAndBytesConfig = None


# -------------------------------------------------------------
# 1. BigQuery Query Builder
# -------------------------------------------------------------
def build_bigquery_sql(query_limit: Optional[int] = None) -> str:
    """Builds the SQL query pulling trending search queries from Google Trends public data."""
    limit_clause = f"LIMIT {int(query_limit)}" if query_limit and query_limit > 0 else ""
    return f"""WITH RankedTerms AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY term
      ORDER BY RAND()
    ) AS term_row_num
  FROM `bigquery-public-data.google_trends.top_terms`
  WHERE week >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
    AND dma_name IS NOT NULL
    AND TRIM(dma_name) != ''
)
SELECT
  dma_name,
  term,
  rank,
  score,
  week
FROM RankedTerms
WHERE term_row_num = 1
ORDER BY dma_name, rank ASC
{limit_clause}"""


# -------------------------------------------------------------
# 2. Stage 1: Preprocess/Postprocess for Gemini Text Generation
# -------------------------------------------------------------
GEMINI_TEXT_INFERENCE_ARGS: Dict[str, Any] = {
    "config": {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "summary": {"type": "STRING"},
                "image_prompt": {"type": "STRING"},
            },
            "required": ["summary", "image_prompt"],
        },
        "temperature": 0.7,
    }
}


def preprocess_for_gemini_text(
    city_terms_pair: Tuple[str, Iterable[str]]
) -> Tuple[Dict[str, Any], str]:
    """Formats city search terms into a prompt and carries context in the key."""
    city, terms = city_terms_pair
    terms_list = list(terms)[:10]
    terms_text = "\n".join([f"- {t}" for t in terms_list])

    prompt = f"""You are a visionary cultural anthropologist and art director.
Here are the top trending search topics in "{city}" for the week:
{terms_text}

Synthesize these trends and respond with valid JSON containing exactly two keys:
1. "summary": A vibrant, punchy 2-sentence cultural summary capturing the city's current pulse, mood, and obsessions.
2. "image_prompt": A vivid, photorealistic cinematic art prompt describing a captivating visual scene that embodies the spirit and trends of {city}. Do not include any text, letters, or logos in the image prompt."""
    context = {"city": city, "terms": terms_list}
    return (context, prompt)


def postprocess_gemini_text(
    element: Tuple[Dict[str, Any], PredictionResult]
) -> Dict[str, Any]:
    """Extracts structured JSON summary & image_prompt and enriches the city context."""
    context, pred = element
    city = context["city"]
    raw_text = ""
    try:
        response = pred.inference
        if isinstance(response, str):
            raw_text = response.strip()
        elif hasattr(response, "text") and response.text:
            raw_text = response.text.strip()
        elif hasattr(response, "parts") and response.parts:
            raw_text = "".join(
                p.text for p in response.parts if getattr(p, "text", None)
            ).strip()
        elif hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            raw_text = "".join(
                p.text for p in parts if getattr(p, "text", None)
            ).strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        summary = parsed.get("summary", "Trending updates.")
        image_prompt = parsed.get("image_prompt", f"Scenic cityscape of {city}")
    except Exception as e:
        logging.warning("Gemini output parse notice for %s (%s)", city, e)
        summary = (
            raw_text[:200]
            if raw_text
            else f"{city} is buzzing with cultural events, live entertainment, and trending local activities."
        )
        image_prompt = f"Atmospheric cinematic cityscape of {city} during golden hour"

    return {**context, "summary": summary, "image_prompt": image_prompt}


# -------------------------------------------------------------
# 3. Stage 2: Preprocess/Postprocess for Gemini Image Generation
# -------------------------------------------------------------
def _render_fallback_card(prompt: str) -> bytes:
    """Renders a fallback PNG card if inline image generation data is unavailable."""
    img = Image.new("RGB", (768, 768), color=(26, 32, 44))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (768, 14)], fill=(66, 133, 244))
    draw.rectangle([(0, 14), (768, 22)], fill=(234, 67, 53))
    draw.text((40, 50), "GOOGLE TRENDS x GEMINI BULK INFERENCE", fill=(160, 174, 192))
    draw.text((40, 90), "City Trend Art Card", fill=(247, 250, 252))

    words = prompt.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > 42:
            lines.append(" ".join(cur[:-1]))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))

    y = 160
    draw.text((40, y), "Artistic Scene Prompt:", fill=(251, 188, 4))
    y += 35
    for line in lines[:10]:
        draw.text((40, y), line, fill=(226, 232, 240))
        y += 26

    draw.text((40, 710), "Powered by Apache Beam RunInference on Dataflow", fill=(128, 134, 139))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def preprocess_for_gemini_image(
    context: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """Extracts the synthesized image_prompt while preserving full city context in the key."""
    return (context, context["image_prompt"])


def postprocess_gemini_image(
    element: Tuple[Dict[str, Any], PredictionResult]
) -> Dict[str, Any]:
    """Extracts generated PNG image bytes from built-in GeminiModelHandler response."""
    context, pred = element
    city = context["city"]
    prompt_str = context["image_prompt"]
    image_bytes = None
    image_source = "gemini-2.5-flash-image"
    try:
        response = pred.inference
        parts = getattr(response, "parts", None)
        if not parts and getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts
        if parts:
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None and getattr(inline_data, "data", None):
                    img = Image.open(io.BytesIO(inline_data.data))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
                    break
    except Exception as e:
        logging.warning("Gemini image extraction notice for %s (%s)", city, e)

    if image_bytes is None:
        image_bytes = _render_fallback_card(prompt_str)
        image_source = "fallback-card-renderer"

    return {**context, "image_bytes": image_bytes, "image_source": image_source}


# -------------------------------------------------------------
# 4. Stage 3: Local Vectorized ShieldGemma 2 ModelHandler
# -------------------------------------------------------------
class _SharedShieldGemma2:
    """Wrapper around ShieldGemma 2 model and processor for large_model=True."""

    DEFAULT_POLICIES = ["dangerous", "sexual", "violence"]

    def __init__(self, model: Any, processor: Any, precision_mode: str = "bfloat16"):
        self.model = model
        self.processor = processor
        self.precision_mode = precision_mode
        self._patch_lm_head_last_token_only()

    def _patch_lm_head_last_token_only(self) -> None:
        """Wraps lm_head so hidden_states are sliced to (B, 1, hidden) before linear projection."""
        try:
            inner = getattr(self.model, "model", None)
            lm_head = getattr(inner, "lm_head", None)
            if lm_head is not None and not getattr(lm_head, "_last_token_patched", False):
                orig_forward = lm_head.forward

                def _sliced_lm_head_forward(hidden_states: Any) -> Any:
                    if hasattr(hidden_states, "ndim") and hidden_states.ndim == 3 and hidden_states.shape[1] > 1:
                        hidden_states = hidden_states[:, -1:, :]
                    return orig_forward(hidden_states)

                lm_head.forward = _sliced_lm_head_forward
                lm_head._last_token_patched = True
        except Exception:
            pass

    def _score_single_image_policies(
        self,
        rgb_image: np.ndarray,
        policies: Sequence[str],
        custom_policies: Optional[Mapping[str, str]],
        safety_threshold: float,
        policy_chunk_size: int,
    ) -> Dict[str, Any]:
        policy_scores: Dict[str, float] = {}
        policies_list = list(policies)

        for p_start in range(0, len(policies_list), policy_chunk_size):
            sub_policies = policies_list[p_start : p_start + policy_chunk_size]
            proc_kwargs: Dict[str, Any] = {
                "images": [rgb_image],
                "policies": sub_policies,
                "return_tensors": "pt",
            }
            if custom_policies:
                proc_kwargs["custom_policies"] = dict(custom_policies)

            inputs = self.processor(**proc_kwargs).to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**inputs, logits_to_keep=1, use_cache=False)
                probs = outputs.probabilities.detach().cpu().float().view(len(sub_policies), 2).tolist()

            del inputs, outputs
            for p_idx, policy_name in enumerate(sub_policies):
                policy_scores[policy_name] = round(float(probs[p_idx][0]), 4)

        violated_policies = [
            policy for policy, score in policy_scores.items() if score >= safety_threshold
        ]
        return {
            "is_safe": len(violated_policies) == 0,
            "violated_policies": violated_policies,
            "policy_scores": policy_scores,
            "threshold": safety_threshold,
            "evaluation_mode": f"shieldgemma-2-{self.precision_mode}",
        }

    def __call__(
        self,
        batch: Sequence[bytes],
        policies: Sequence[str],
        custom_policies: Optional[Mapping[str, str]],
        safety_threshold: float,
    ) -> List[Dict[str, Any]]:
        rgb_images = [
            np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"), copy=True)
            for img_bytes in batch
        ]

        policy_chunk_size = len(policies)
        if torch is not None and torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_vram_gb < 20.0:
                policy_chunk_size = 1

        all_verdicts: List[Dict[str, Any]] = []
        try:
            for rgb_img in rgb_images:
                verdict = self._score_single_image_policies(
                    rgb_img,
                    policies,
                    custom_policies,
                    safety_threshold,
                    policy_chunk_size=policy_chunk_size,
                )
                all_verdicts.append(verdict)
        finally:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_verdicts


class ShieldGemma2ModelHandler(ModelHandler[bytes, PredictionResult, Any]):
    """Vectorized Apache Beam ModelHandler for local ShieldGemma 2 GPU inference."""

    DEFAULT_POLICIES = ["dangerous", "sexual", "violence"]

    def __init__(
        self,
        model_id: str = "google/shieldgemma-2-4b-it",
        hf_token: Optional[str] = None,
        precision: str = "auto",
        safety_threshold: float = 0.5,
        fail_closed: bool = True,
        policies: Optional[Sequence[str]] = None,
        custom_policies: Optional[Mapping[str, str]] = None,
        min_batch_size: int = 1,
        max_batch_size: int = 1,
        large_model: bool = True,
    ):
        self.model_id = model_id
        self.hf_token = hf_token or None
        self.precision = precision
        self.safety_threshold = safety_threshold
        self.fail_closed = fail_closed
        self.policies = list(policies) if policies else list(self.DEFAULT_POLICIES)
        self.custom_policies = dict(custom_policies) if custom_policies else None
        self._large_model = large_model
        self._share_across_processes = large_model
        self._batching_kwargs = {
            "min_batch_size": min_batch_size,
            "max_batch_size": max_batch_size,
        }

    def share_model_across_processes(self) -> bool:
        return self._large_model

    def batch_elements_kwargs(self) -> Mapping[str, Any]:
        return self._batching_kwargs

    def get_resource_hints(self) -> dict:
        return {
            "min_ram": "32GB",
            "accelerator": "type:nvidia-l4;count:1;install-nvidia-driver",
        }

    def load_model(self) -> Optional[_SharedShieldGemma2]:
        if torch is None or AutoProcessor is None or ShieldGemma2ForImageClassification is None:
            logging.warning("PyTorch/Transformers not installed. Skipping local ShieldGemma 2 loading.")
            return None

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            use_4bit = False
            use_8bit = False
            if device == "cuda" and BitsAndBytesConfig is not None:
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if self.precision == "4bit" or (self.precision == "auto" and total_vram_gb < 20.0):
                    use_4bit = True
                elif self.precision == "8bit":
                    use_8bit = True

            processor = AutoProcessor.from_pretrained(
                self.model_id,
                token=self.hf_token,
            )

            load_kwargs: Dict[str, Any] = {
                "torch_dtype": dtype,
                "device_map": {"": 0} if device == "cuda" else None,
                "low_cpu_mem_usage": True,
                "token": self.hf_token,
            }
            precision_label = "bfloat16" if device == "cuda" else "float32"

            if use_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                precision_label = "4bit-nf4-bfloat16"
            elif use_8bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                precision_label = "8bit-bfloat16"

            logging.info("Loading ShieldGemma 2 (%s) on %s (%s)...", self.model_id, device, precision_label)
            try:
                model = ShieldGemma2ForImageClassification.from_pretrained(
                    self.model_id,
                    **load_kwargs,
                )
            except Exception as quant_err:
                if "quantization_config" in load_kwargs:
                    logging.warning("Quantized load fallback to bfloat16 (%s)...", quant_err)
                    load_kwargs.pop("quantization_config", None)
                    precision_label = "bfloat16"
                    model = ShieldGemma2ForImageClassification.from_pretrained(
                        self.model_id,
                        **load_kwargs,
                    )
                else:
                    raise

            if device == "cpu":
                model = model.to(torch.device("cpu"))
            model.eval()
            logging.info("✅ ShieldGemma 2 loaded successfully on %s (%s)!", model.device, precision_label)
            return _SharedShieldGemma2(
                model=model,
                processor=processor,
                precision_mode=precision_label,
            )
        except Exception as e:
            logging.warning(
                "ShieldGemma 2 model load notice (%s). "
                "Ensure you accepted terms on Hugging Face and provided a valid token.",
                e,
            )
            return None

    def run_inference(
        self,
        batch: Sequence[bytes],
        model: Any,
        inference_args: Optional[Dict[str, Any]] = None,
    ) -> Iterable[PredictionResult]:
        if not batch:
            return []

        if model is not None:
            try:
                verdicts = model(
                    batch,
                    self.policies,
                    self.custom_policies,
                    self.safety_threshold,
                )
                return [
                    PredictionResult(img_bytes, verdict, self.model_id)
                    for img_bytes, verdict in zip(batch, verdicts)
                ]
            except Exception as e:
                logging.error("ShieldGemma 2 batched inference error: %s", e)
                if self.fail_closed:
                    raise
            finally:
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        unverified_is_safe = not self.fail_closed
        unverified_violations = ["unverified_model_unavailable"] if self.fail_closed else []
        unverified_scores = {policy: None for policy in self.policies}
        return [
            PredictionResult(
                img_bytes,
                {
                    "is_safe": unverified_is_safe,
                    "violated_policies": unverified_violations,
                    "policy_scores": unverified_scores,
                    "threshold": self.safety_threshold,
                    "evaluation_mode": "unverified-fail-closed" if self.fail_closed else "unverified-demo-fallback",
                },
                self.model_id,
            )
            for img_bytes in batch
        ]


def preprocess_for_shieldgemma(
    context: Dict[str, Any]
) -> Tuple[Dict[str, Any], bytes]:
    """Extracts image_bytes for ShieldGemma 2 while preserving city context in the key."""
    return (context, context["image_bytes"])


def postprocess_shieldgemma(
    element: Tuple[Dict[str, Any], PredictionResult]
) -> Tuple[str, Dict[str, Any]]:
    """Attaches ShieldGemma 2 safety verdict to the final city record."""
    context, pred = element
    return (context["city"], {**context, "safety_check": pred.inference})


# -------------------------------------------------------------
# 5. Routing, Sinks & DLQ
# -------------------------------------------------------------
class RouteBySafetyDoFn(beam.DoFn):
    """Routes records to 'safe' (main output) or 'unsafe' (tagged output)."""

    UNSAFE_TAG = "unsafe"

    def process(self, element: Tuple[str, Dict[str, Any]]):
        city, data = element
        safety_check = data.get("safety_check", {})
        is_safe = safety_check.get("is_safe", False)
        if is_safe:
            yield (city, data)
        else:
            yield beam.pvalue.TaggedOutput(self.UNSAFE_TAG, (city, data))


class SaveOutputsToGCSDoFn(beam.DoFn):
    """Writes images and metadata JSON to target storage bucket."""

    def __init__(self, output_path: str, bucket_label: str = "safe"):
        self.output_path = output_path.rstrip("/")
        self.bucket_label = bucket_label

    def process(self, element: Tuple[str, Dict[str, Any]]):
        city, data = element
        clean_city_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", city).strip("_")

        image_dest = f"{self.output_path}/images/{clean_city_slug}.png"
        meta_dest = f"{self.output_path}/metadata/{clean_city_slug}.json"

        image_bytes = data.get("image_bytes")
        if image_bytes:
            with FileSystems.create(image_dest) as f:
                f.write(image_bytes)

        payload = {
            "city": city,
            "status": self.bucket_label,
            "terms": data.get("terms", []),
            "summary": data.get("summary"),
            "image_prompt": data.get("image_prompt"),
            "image_source": data.get("image_source"),
            "image_gcs_uri": image_dest,
            "safety_check": data.get("safety_check", {}),
        }
        with FileSystems.create(meta_dest) as f:
            f.write(json.dumps(payload, indent=2).encode("utf-8"))

        yield (city, self.bucket_label, image_dest, meta_dest)


class SaveDLQErrorsToGCSDoFn(beam.DoFn):
    """Writes failed inference batches/elements from RunInferenceDLQ to Quarantine storage."""

    def __init__(self, dlq_output_path: str):
        self.dlq_output_path = dlq_output_path.rstrip("/")

    def process(self, failed_element: Tuple[Any, Any]):
        failed_input, error_msg = failed_element
        record_id = uuid.uuid4().hex[:10]
        dlq_dest = f"{self.dlq_output_path}/error_{record_id}.json"

        saved_dlq_images = []
        try:
            items = failed_input if isinstance(failed_input, (list, tuple)) else [failed_input]
            for item in items:
                ctx = (
                    item[0]
                    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict)
                    else (item if isinstance(item, dict) else None)
                )
                if isinstance(ctx, dict) and ctx.get("image_bytes"):
                    city_name = ctx.get("city", record_id)
                    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(city_name)).strip("_")
                    img_uri = f"{self.dlq_output_path}/images/{slug}_{record_id}.png"
                    with FileSystems.create(img_uri) as img_f:
                        img_f.write(ctx["image_bytes"])
                    saved_dlq_images.append(img_uri)
        except Exception:
            pass

        payload = {
            "record_id": record_id,
            "error": str(error_msg),
            "saved_dlq_images": saved_dlq_images,
            "failed_input_repr": repr(failed_input)[:500],
        }
        with FileSystems.create(dlq_dest) as f:
            f.write(json.dumps(payload, indent=2).encode("utf-8"))
        yield dlq_dest


# -------------------------------------------------------------
# 6. Main Pipeline Runner
# -------------------------------------------------------------
def run(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Apache Beam Bulk ML Inference with Gemini & ShieldGemma 2 on Dataflow."
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        required=True,
        help="GCS bucket URI for safe/approved outputs (e.g. gs://YOUR_BUCKET/safe_output)",
    )
    parser.add_argument(
        "--unsafe_output_path",
        dest="unsafe_output_path",
        required=True,
        help="GCS bucket URI for quarantine outputs (e.g. gs://YOUR_BUCKET/quarantine_output)",
    )
    parser.add_argument(
        "--hf_token",
        dest="hf_token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face read token for google/shieldgemma-2-4b-it",
    )
    parser.add_argument(
        "--gemini_text_model",
        dest="gemini_text_model",
        default="gemini-2.5-flash",
        help="Gemini model for cultural summary and image prompt synthesis",
    )
    parser.add_argument(
        "--gemini_image_model",
        dest="gemini_image_model",
        default="gemini-2.5-flash-image",
        help="Gemini model for image generation",
    )
    parser.add_argument(
        "--use_vertex_flex_api",
        dest="use_vertex_flex_api",
        action="store_true",
        default=False,
        help="Whether to route Gemini requests through the Vertex AI Flex API",
    )
    parser.add_argument(
        "--shieldgemma_model",
        dest="shieldgemma_model",
        default="google/shieldgemma-2-4b-it",
        help="Hugging Face model ID for ShieldGemma 2",
    )
    parser.add_argument(
        "--shieldgemma_precision",
        dest="shieldgemma_precision",
        default="auto",
        choices=["auto", "4bit", "8bit", "bfloat16"],
        help="Weight precision for ShieldGemma 2",
    )
    parser.add_argument(
        "--safety_threshold",
        dest="safety_threshold",
        type=float,
        default=0.5,
        help="Probability threshold to flag a policy violation (0.0 - 1.0)",
    )
    parser.add_argument(
        "--fail_closed",
        dest="fail_closed",
        action="store_true",
        default=True,
        help="Route unverified/failed safety checks to quarantine bucket",
    )
    parser.add_argument(
        "--query_limit",
        dest="query_limit",
        type=int,
        default=None,
        help="Optional row limit on BigQuery Google Trends source query (useful for smoke tests)",
    )

    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_out_dir = f"{known_args.output_path.rstrip('/')}/trends_inference_{run_ts}"
    unsafe_out_dir = f"{known_args.unsafe_output_path.rstrip('/')}/trends_inference_{run_ts}"
    dlq_out_dir = f"{unsafe_out_dir}/dlq"

    logging.info("Safe destination output:       %s", safe_out_dir)
    logging.info("Quarantine destination output: %s", unsafe_out_dir)
    logging.info("Dead-Letter Queue path:        %s", dlq_out_dir)

    bigquery_sql = build_bigquery_sql(known_args.query_limit)

    _gemini_extra_kwargs = (
        {"use_vertex_flex_api": known_args.use_vertex_flex_api}
        if "use_vertex_flex_api" in inspect.signature(GeminiModelHandler.__init__).parameters
        else {}
    )

    gemini_text_handler = (
        KeyedModelHandler(
            GeminiModelHandler(
                model_name=known_args.gemini_text_model,
                request_fn=generate_from_string,
                max_batch_size=1,
                **_gemini_extra_kwargs,
            )
        )
        .with_preprocess_fn(preprocess_for_gemini_text)
        .with_postprocess_fn(postprocess_gemini_text)
    )

    gemini_image_handler = (
        KeyedModelHandler(
            GeminiModelHandler(
                model_name=known_args.gemini_image_model,
                request_fn=generate_image_from_strings_and_images,
                max_batch_size=1,
                **_gemini_extra_kwargs,
            )
        )
        .with_preprocess_fn(preprocess_for_gemini_image)
        .with_postprocess_fn(postprocess_gemini_image)
    )

    shieldgemma_handler = (
        KeyedModelHandler(
            ShieldGemma2ModelHandler(
                model_id=known_args.shieldgemma_model,
                hf_token=known_args.hf_token,
                precision=known_args.shieldgemma_precision,
                safety_threshold=known_args.safety_threshold,
                fail_closed=known_args.fail_closed,
                max_batch_size=1,
                large_model=True,
            )
        )
        .with_preprocess_fn(preprocess_for_shieldgemma)
        .with_postprocess_fn(postprocess_shieldgemma)
    )

    with beam.Pipeline(options=pipeline_options) as p:
        # Step 1: BigQuery Google Trends with pre-shuffle combiner lifting
        sampled_city_terms = (
            p
            | "ReadFromBigQuery" >> beam.io.ReadFromBigQuery(
                query=bigquery_sql,
                use_standard_sql=True,
            )
            | "FilterValidDMA" >> beam.Filter(
                lambda row: row.get("dma_name") is not None and str(row.get("dma_name", "")).strip() != ""
            )
            | "ExtractCityTerm" >> beam.Map(
                lambda row: (
                    str(row["dma_name"]).strip(),
                    f"{row.get('term', '')} (Rank: {row.get('rank', 'N/A')}, Score: {row.get('score', 'N/A')})",
                )
            )
            | "SampleTopTermsPerCity" >> beam.combiners.Sample.FixedSizePerKey(10)
        )

        # Step 2: RunInference Stage 1 - Built-in Gemini Text Generation (Structured JSON)
        gemini_text_records = (
            sampled_city_terms
            | "RunGeminiTextInference" >> RunInference(
                gemini_text_handler,
                inference_args=GEMINI_TEXT_INFERENCE_ARGS,
            )
        )

        # Step 3: RunInference Stage 2 - Built-in Gemini Image Generation (Zero-Shuffle)
        gemini_image_records = (
            gemini_text_records
            | "RunGeminiImageInference" >> RunInference(gemini_image_handler)
        )

        # Step 4: RunInference Stage 3 - Local GPU ShieldGemma 2 Safety Moderation + DLQ
        moderated_records, safety_dlq = (
            gemini_image_records
            | "RunShieldGemma2SafetyCheck" >> RunInference(
                shieldgemma_handler
            ).with_exception_handling()
        )

        # Step 5: Route Safe vs. Unsafe Records
        routed_records = (
            moderated_records
            | "RouteByShieldGemmaSafety" >> beam.ParDo(RouteBySafetyDoFn()).with_outputs(
                RouteBySafetyDoFn.UNSAFE_TAG, main="safe"
            )
        )

        # Step 6a: Sink Safe records to Primary GCS
        _ = routed_records.safe | "SaveSafeToGCS" >> beam.ParDo(
            SaveOutputsToGCSDoFn(safe_out_dir, bucket_label="safe")
        )

        # Step 6b: Sink Unsafe/Quarantine records to Quarantine GCS
        _ = routed_records[RouteBySafetyDoFn.UNSAFE_TAG] | "SaveUnsafeToGCS" >> beam.ParDo(
            SaveOutputsToGCSDoFn(unsafe_out_dir, bucket_label="unsafe_quarantine")
        )

        # Step 6c: Sink DLQ exceptions to Quarantine DLQ
        _ = safety_dlq.failed_inferences | "SaveSafetyDLQToGCS" >> beam.ParDo(
            SaveDLQErrorsToGCSDoFn(dlq_out_dir)
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run(sys.argv[1:])
