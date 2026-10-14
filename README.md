# Apache Beam Bulk ML Inference with Gemini & ShieldGemma 2 on Dataflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashwin-sampathkumar/ApacheBeam-Dataflow-Demos/blob/main/notebooks/beam_bulk_ml_inference_gemini_shieldgemma.ipynb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-brightgreen.svg)](https://www.python.org/)
[![Apache Beam](https://img.shields.io/badge/Apache%20Beam-2.59%2B-orange.svg)](https://beam.apache.org/)
[![Google Cloud Dataflow](https://img.shields.io/badge/Google%20Cloud-Dataflow-4285F4.svg)](https://cloud.google.com/dataflow)

This repository contains a production-grade **Apache Beam Bulk ML Inference (`RunInference`)** demonstration that chains **remote Gemini foundation models** (via Beam's built-in `GeminiModelHandler`) with **local GPU safety moderation** using **ShieldGemma 2** (`google/shieldgemma-2-4b-it`) on **Google Cloud Dataflow**.

---

## 🏗️ Architecture: Zero-Shuffle Linear Inference Chain

```text
┌────────────────────────────────────────────────────────┐
│ BigQuery Source: Google Trends Public Dataset          │  SELECT * FROM `bigquery-public-data.google_trends.top_terms`
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ beam.combiners.Sample.FixedSizePerKey(10)              │  Pre-shuffle combiner lifting (top 10 terms per dma_name)
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ RunInference Stage 1: Built-in GeminiModelHandler      │  generate_from_string + JSON response_schema + Flex API
│ + with_preprocess_fn / with_postprocess_fn             │  Emits: {city, terms, summary, image_prompt}
└───────────────────────────┬────────────────────────────┘
                            │  (No CoGroupByKey shuffle!)
                            ▼
┌────────────────────────────────────────────────────────┐
│ RunInference Stage 2: Built-in GeminiModelHandler      │  generate_image_from_strings_and_images (gemini-2.5-flash-image)
│ + with_preprocess_fn / with_postprocess_fn             │  Emits: {city, summary, image_prompt, image_bytes}
└───────────────────────────┬────────────────────────────┘
                            │  (No CoGroupByKey shuffle!)
                            ▼
┌────────────────────────────────────────────────────────┐
│ RunInference Stage 3: Local ShieldGemma 2 (Vectorized) │  ShieldGemma2ForImageClassification (B*P batched forward pass)
│ + Right-Fitting GPU Resource Hints + .with_exception_  │  Evaluates: dangerous, sexual, violence policies
│   handling() (RunInferenceDLQ)                         │
└─────────────┬─────────────────────────────┬────────────┘
              │                             │ Exceptions (DLQ)
              ▼                             ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│ ✅ Approved GCS Bucket     │ │ 🚨 Quarantine GCS Bucket  │ │ 🛑 Quarantine DLQ Path    │
│  (Passed ShieldGemma 2)   │ │  (Policy Violations)      │ │  (Inference Exceptions)   │
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
```

---

## 💡 Production Dataflow & Apache Beam Best Practices

This project incorporates architectural recommendations for production ML pipelines on Apache Beam and Cloud Dataflow:

1. **Pre-Shuffle Combiner Lifting (`Sample.FixedSizePerKey(10)`)**:
   Instead of using `GroupByKey()` to collect all rows per Designated Market Area (DMA) before slicing top terms, we use `beam.combiners.Sample.FixedSizePerKey(10)` so mapper workers aggregate top terms locally *before* shuffle, reducing network transfer by orders of magnitude on large BigQuery tables.

2. **Zero-Shuffle Linear Chaining (`with_preprocess_fn` / `with_postprocess_fn`)**:
   Standard multi-stage ML pipelines often fork into separate PCollections and join results back using `CoGroupByKey()`. This forces high-resolution generated images (hundreds of KB to megabytes each) across the Dataflow shuffle service. By chaining stages sequentially using Beam's `.with_preprocess_fn()` and `.with_postprocess_fn()`, element metadata is preserved through composite keys with zero intermediate network shuffle.

3. **Built-in `GeminiModelHandler` with Structured Outputs & Flex API**:
   Uses Beam's official `apache_beam.ml.inference.gemini_inference.GeminiModelHandler` with Vertex AI Structured Outputs (`response_schema`), guaranteeing strict JSON formatting directly from the model without fragile regex or string parsing. Also supports the **Vertex Flex API** (`use_vertex_flex_api=True`) for high-throughput batch workloads.

4. **Local Vectorized GPU Safety Moderation (`ShieldGemma 2`)**:
   Runs Google's **`google/shieldgemma-2-4b-it`** (`ShieldGemma2ForImageClassification`) locally on worker GPUs with `large_model=True` and `MultiProcessShared` to share loaded model weights across worker threads. Employs `torch.bfloat16` precision and explicit post-inference memory reclamation (`gc.collect()` + `torch.cuda.empty_cache()`) to prevent GPU CUDA OOMs.

5. **Dataflow Right-Fitting GPU Resource Hints**:
   Decorates the `ShieldGemma 2` transform with `min_ram="32GB"` and `accelerator="type:nvidia-l4;count:1;install-nvidia-driver"` so expensive GPU worker machines are only allocated to the vision safety stage, while CPU-only transforms (BigQuery read, Gemini API calls, GCS sinks) execute on standard CPU workers.

6. **Native `RunInference` Dead-Letter Queue (`RunInferenceDLQ`)**:
   Employs `.with_exception_handling()` on `RunInference` to safely catch model inference exceptions (such as corrupted images, transient CUDA faults, or rate limits) into a dedicated DLQ PCollection without failing the overall pipeline.

7. **Fail-Closed Conditional GCS Routing**:
   If an image violates safety policies or cannot be verified (e.g., missing model token), it is strictly routed to the **Quarantine GCS Bucket** rather than failing open into production.

---

## 📁 Repository Structure

```
ApacheBeam-Dataflow-Demos/
├── README.md                      # Project documentation and architecture guide
├── LICENSE                        # Apache 2.0 License
├── .gitignore                     # Git ignore rules for Python, Beam, and Jupyter
├── notebooks/
│   └── beam_bulk_ml_inference_gemini_shieldgemma.ipynb   # Production Google Colab notebook
└── pipeline/                      # Modular production script for headless execution
    ├── __init__.py
    ├── main.py                    # Main pipeline entrypoint with CLI arguments
    └── requirements.txt           # Worker dependencies for Dataflow and local environments
```

---

## 🔑 Prerequisites & Setup

### 1. Google Cloud Platform (GCP)
Ensure you have an active GCP project with the following APIs enabled:
* **Cloud Dataflow API** (`dataflow.googleapis.com`)
* **Vertex AI API** (`aiplatform.googleapis.com`)
* **BigQuery API** (`bigquery.googleapis.com`)
* **Google Cloud Storage** (`storage.googleapis.com`)

Create two Google Cloud Storage buckets (or folders within a bucket):
1. **Primary / Safe Bucket**: `gs://<YOUR_BUCKET_NAME>/bulk_inference_demo/safe_output`
2. **Quarantine Bucket**: `gs://<YOUR_BUCKET_NAME>/bulk_inference_demo/quarantine_output`

### 2. Hugging Face Gated Model Access (`ShieldGemma 2`)
ShieldGemma 2 is a gated model:
1. Visit [Hugging Face: google/shieldgemma-2-4b-it](https://huggingface.co/google/shieldgemma-2-4b-it) and accept the terms of use.
2. Generate a Read Token at [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens).
3. If running in Google Colab, store this token in **Colab Secrets** (the 🔑 icon) as `HF_TOKEN`. If running via CLI, set the environment variable:
   ```bash
   export HF_TOKEN="your_hugging_face_read_token"
   ```

---

## 🚀 Getting Started

### Option A: Google Colab (Interactive)
The easiest way to explore this pipeline interactively is via Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashwin-sampathkumar/ApacheBeam-Dataflow-Demos/blob/main/notebooks/beam_bulk_ml_inference_gemini_shieldgemma.ipynb)

1. Open the notebook in Colab.
2. Select a GPU runtime (**Runtime > Change runtime type > T4 or L4 GPU**).
3. Set your `PROJECT_ID`, `GCS_BUCKET`, `GCS_UNSAFE_BUCKET`, and `HF_TOKEN` in Step 2.
4. Run all cells to execute the pipeline and view the visual safety audit in Step 6.

---

### Option B: Command Line (Modular Pipeline)

#### 1. Local Smoke Test (`DirectRunner`)
Run a quick end-to-end smoke test on a small subset of BigQuery Google Trends data (`--query_limit=5`) locally on your machine:

```bash
# Clone the repository
git clone https://github.com/ashwin-sampathkumar/ApacheBeam-Dataflow-Demos.git
cd ApacheBeam-Dataflow-Demos

# Install dependencies
pip install -r pipeline/requirements.txt

# Run locally with DirectRunner
python -m pipeline.main \
    --runner=DirectRunner \
    --project=YOUR_GCP_PROJECT_ID \
    --output_path=gs://YOUR_BUCKET_NAME/safe_output \
    --unsafe_output_path=gs://YOUR_BUCKET_NAME/quarantine_output \
    --query_limit=5
```

#### 2. Cloud Dataflow Distributed Execution (`DataflowRunner`)
Submit the full distributed bulk ML pipeline to Google Cloud Dataflow with Right-Fitting L4 GPU worker accelerators:

```bash
python -m pipeline.main \
    --runner=DataflowRunner \
    --project=YOUR_GCP_PROJECT_ID \
    --region=us-central1 \
    --temp_location=gs://YOUR_BUCKET_NAME/temp \
    --staging_location=gs://YOUR_BUCKET_NAME/staging \
    --output_path=gs://YOUR_BUCKET_NAME/bulk_inference_demo/safe_output \
    --unsafe_output_path=gs://YOUR_BUCKET_NAME/bulk_inference_demo/quarantine_output \
    --requirements_file=pipeline/requirements.txt \
    --machine_type=g2-standard-8 \
    --dataflow_service_options="worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver" \
    --experiments=use_runner_v2,enable_recommendation_service \
    --job_name="dataflow-gemini-shieldgemma-trends-$(date +%Y%m%d-%H%M%S)"
```

---

## 📊 Safety Evaluation & Visual Audit

The pipeline evaluates each generated image against three core safety policies:
* **`dangerous`**: Content encouraging or depicting hazardous, self-harm, or harmful acts.
* **`sexual`**: Sexually explicit imagery.
* **`violence`**: Graphic or severe violence.

Each output JSON record in GCS includes full evaluation metadata:
```json
{
  "city": "Austin, TX",
  "status": "safe",
  "terms": ["Austin City Limits Music Festival", "South by Southwest", "..."],
  "summary": "Austin pulses with live indie music, food-truck culture, and tech innovation...",
  "image_prompt": "A warm cinematic twilight view over Lady Bird Lake with festival lights in the background...",
  "image_source": "gemini-2.5-flash-image",
  "image_gcs_uri": "gs://YOUR_BUCKET/safe_output/images/Austin_TX.png",
  "safety_check": {
    "is_safe": true,
    "violated_policies": [],
    "policy_scores": {
      "dangerous": 0.0012,
      "sexual": 0.0004,
      "violence": 0.0008
    },
    "threshold": 0.5,
    "evaluation_mode": "shieldgemma-2-bfloat16"
  }
}
```

---

## 💰 Cost & Quota Notes

* **BigQuery Public Data**: The Google Trends public dataset is queried with partition filtering on `week >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)` to keep bytes scanned minimal.
* **Vertex AI Gemini Quotas**: If running large-scale bulk jobs, consider setting `--use_vertex_flex_api` to route requests through the Flex API for lower cost and batch-optimized quotas.
* **Dataflow GPUs**: Right-Fitting ensures GPU charges only accrue for the `RunShieldGemma2SafetyCheck` stage on `nvidia-l4` accelerators (`g2-standard-8`), while preprocessing and I/O transforms scale on standard CPU instances.

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
