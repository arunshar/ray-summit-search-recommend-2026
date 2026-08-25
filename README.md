# Real-Time Search & Recommendation Systems for AI Commerce

Training materials for **Ray Summit 2026**. This course walks through batch inference, online serving, and periodic model updates for an e-commerce recommender plus semantic search stack built on Ray.

You start from a pre-trained two-tower recommender and a product catalog with description embeddings, then:

1. Run **batch recommendation** over a user dataset with Ray Data.
2. Serve **live recommendations and search** with composed Ray Serve deployments.
3. **Fine-tune** the recommender on fresh interaction data with Ray Train.
4. Optionally productionize the service on **Anyscale**.

## Provenance

This is Ray Summit 2026 Anyscale course material, republished with permission for personal study. Original content copyright 2026 Anyscale.

## What you will build

```
                    HTTP POST /
                         │
                      Ingress
                    /         \
           Recommender      SemanticSearch
                    \         /
                  DatabaseFacade
                         │
              users.ndjson + catalog parquet
```

- **Recommender** — two-tower PyTorch model (`TwoTower` in `recommend.py`) that returns top-k items per user.
- **SemanticSearch** — embeds the query with Sentence Transformers, scores against precomputed product description embeddings, then re-ranks by string distance.
- **Ingress** — always returns recommendations; adds search results when the request includes a `query`.
- **DatabaseFacade** — shared lookup of users and products.

A typical request body:

```json
{
  "id": "7034dd99-ceb3-474d-a0ba-5beaf122273f",
  "query": "steel bowls"
}
```

Response shape:

```json
{
  "recommendations": { "...": "..." },
  "search_results": { "...": "..." }
}
```

Omit `query` to get recommendations only.

## Repository layout

| Path | Purpose |
| --- | --- |
| `1_Inference.ipynb` | Session 1: batch inference with Ray Data, then online serving with Ray Serve |
| `2_Training.ipynb` | Session 2: baseline PyTorch training, port to Ray Train, Ray Data + Train, scheduled jobs |
| `recommend.py` | `TwoTower` model and `Recommend` helper used by batch and online paths |
| `search_and_recommend.py` | Composed Serve graph: Ingress, Recommender, SemanticSearch, DatabaseFacade |
| `serve_config.yaml` | Local `serve build` config (import path `search_and_recommend:bound_ingress`) |
| `serve_updated.yaml` | Scaled Serve config with replica counts and a packaged `working_dir` |
| `services_demo/` | Anyscale Services demo notebook and production-oriented Serve config |
| `bonus/labs/` | Hands-on labs (fundamentals and AI-assisted) |
| `bonus/solutions/` | Worked solutions for those labs |
| `offline_slides/` | Course slide deck (HTML + PDF) |
| `outline-compute.md` | Full session outline and planned cluster size |
| `Dockerfile` | Image with Ray, PyTorch, sentence-transformers, and related deps |
| `requirements.txt` | Pip pins mirrored from the Dockerfile, for local work outside the container. |

## Suggested path through the material

1. Skim `outline-compute.md` for the session map.
2. Work through `1_Inference.ipynb` (batch pipeline → single Serve deployment → composed graph → CLI deploy).
3. Optional: `services_demo/1_Service.ipynb` for Anyscale canary rollouts.
4. Work through `2_Training.ipynb` (local PyTorch → TorchTrainer → Ray Data shards → jobs).
5. Practice with the labs, using `bonus/solutions/` only after you have tried them.

### Labs

| Lab | Notebook | Focus |
| --- | --- | --- |
| 1 | `bonus/labs/Lab_1_RayServe_Fundamentals.ipynb` | First `@serve.deployment`, bind/run, HTTP query, `num_replicas` |
| 2 | `bonus/labs/Lab_2_RayServe_AIAssisted.ipynb` | Compose recommend + search into a multi-deployment service |
| 3 | `bonus/labs/Lab_3_RayTrain_Fundamentals.ipynb` | Port a pure PyTorch loop to Ray Train |
| 4 | `bonus/labs/Lab_4_RayTrain_AIAssisted.ipynb` | Feed a Ray Data pipeline into Ray Train |

## Running locally

These notebooks expect an Anyscale / Ray cluster layout with data under `/mnt/cluster_storage/ecom/` (users, catalog with embeddings, and a saved two-tower checkpoint). The inference notebook downloads public sample data from `s3://anyscale-public-materials-use2/ecom/` on first run.

### Tested configuration

The canonical environment is the container in `Dockerfile`. Every value below is read from
`Dockerfile` or `requirements.txt`, not measured on a run.

| Component | Version | Where the value comes from |
| --- | --- | --- |
| Base image | `anyscale/ray-llm:2.55.1-py311-cu128` | `Dockerfile`, `FROM` line |
| Ray | 2.55.1 | the `2.55.1` in the image tag, and `ray[serve]==2.55.1` in `requirements.txt` |
| Python | 3.11 | the `py311` in the image tag |
| CUDA | 12.8 | the `cu128` in the image tag |
| PyTorch | 2.10.0 | `Dockerfile` pip pin |
| torchvision | 0.25.0 | `Dockerfile` pip pin |
| transformers | 4.57.4 | `Dockerfile` pip pin |
| sentence-transformers | 5.2.2 | `Dockerfile` pip pin |
| textdistance | 4.6.3 | `Dockerfile` pip pin |
| pyarrow | 19.0.1 | `Dockerfile` pip pin |
| pandas | 3.0.3 | `Dockerfile` pip pin |
| scikit-learn | 1.6.1 | `Dockerfile` pip pin |
| Python recorded in the course notebooks | 3.11.11 | `language_info` in `1_Inference.ipynb` and `2_Training.ipynb` |
| Cluster shape | m5.2xlarge head plus 2 m5.2xlarge workers | `outline-compute.md`, which labels it *Planned Compute* |

Two cautions on that table. The cluster shape is what the outline planned, not a record of what
ran. No file in this repo records a version banner from a cluster run, so the only version a run
actually left behind is the Python 3.11.11 in the notebook metadata.

The two lab notebooks whose outputs are committed here were run outside that image, on a laptop:

| Component | Version |
| --- | --- |
| OS | macOS 26.7, Apple Silicon (arm64), CPU only |
| Python | 3.11.15 |
| Ray | 2.55.1 |
| PyTorch | 2.13.0 |
| nbconvert | 7.17.1 |

That local set is not the pinned set. PyTorch is 2.13.0 against a pin of 2.10.0, and pandas is
3.0.5 against a pin of 3.0.3. Labs 1 and 3 need only Ray core and PyTorch, so the drift does not
affect them, but it does mean the pinned combination in `requirements.txt` has never been
installed or resolved as a set here.

### Serve from the CLI

```bash
# generate or inspect a config
serve build search_and_recommend:bound_ingress -o serve_config.yaml

# deploy
serve deploy serve_config.yaml
```

`serve_updated.yaml` is a thicker config: 4 DatabaseFacade replicas, 2 Recommender replicas, and autoscaling Search/Ingress. It also sets a remote `working_dir` zip so workers do not depend on your workspace files.

Query a running service. The module Ingress in `search_and_recommend.py` runs
`json.loads(await request.json())`, a double parse, so it expects a JSON *string* body.
The Ingress classes defined inline in `1_Inference.ipynb` parse once and take a normal
JSON object instead:

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '"{\"id\":\"7034dd99-ceb3-474d-a0ba-5beaf122273f\",\"query\":\"steel bowls\"}"'
```

### Anyscale Services

See `services_demo/anyscale_serve_config.yaml` and `services_demo/1_Service.ipynb`. That config adds service name, image URI, compute config, cloud, working dir, and query auth.

```bash
anyscale service deploy --config-file services_demo/anyscale_serve_config.yaml
anyscale service status --name search_recommend_c26_61
```

Replace the service URL and query token in the demo notebook with values from `anyscale service status` (do not commit live tokens). During a canary rollout you can pin traffic with the `X-ANYSCALE-VERSION` header (`primary`, `canary`, or a version id).

## Running outside Anyscale

The code expects a course-cluster data layout under `/mnt/cluster_storage/ecom/`: `users.ndjson`, the `cat_with_embeddings` parquet directory, an `hf_cache` snapshot of the embedding model, and the recommender checkpoint at `recommender/base_model/model.pt`. The public data source is `s3://anyscale-public-materials-use2/ecom` (unsigned reads work), which the inference notebook pulls on first run.

`serve_updated.yaml` and `services_demo/` reference artifacts from the course's Anyscale account: the image `anyscale/image/c26:5`, the compute config `head-2a10g-small:2`, and the `education-us-west-2` cloud. Outsiders cannot use those, so the Anyscale deploy steps will not run as written outside that account. The `working_dir` zips point at the `anyscale-materials` S3 bucket, which is publicly readable at the time of writing but not under this repo's control.

Notebook outputs differ from file to file, and two of the labs now carry outputs from a local run. The per-notebook breakdown is in [Committed notebook state](#committed-notebook-state) below.

## Committed notebook state

Counts below come from parsing each `.ipynb`. *Executed* means the code cell carries a non-null
`execution_count`. *With output* means it has at least one saved output. No notebook in the repo
carries a saved error output.

| Notebook | Code cells | Executed | With output | Ships executed |
| --- | ---: | ---: | ---: | --- |
| `1_Inference.ipynb` | 67 | 64 | 46 | Yes, from the course cluster |
| `2_Training.ipynb` | 19 | 18 | 11 | Yes, from the course cluster |
| `bonus/labs/Lab_1_RayServe_Fundamentals.ipynb` | 3 | 3 | 1 | Yes, from a local CPU run |
| `bonus/labs/Lab_2_RayServe_AIAssisted.ipynb` | 5 | 0 | 0 | No |
| `bonus/labs/Lab_3_RayTrain_Fundamentals.ipynb` | 4 | 4 | 2 | Yes, from a local CPU run |
| `bonus/labs/Lab_4_RayTrain_AIAssisted.ipynb` | 6 | 2 | 0 | No |
| `bonus/solutions/Lab_1_solution.ipynb` | 5 | 0 | 0 | No |
| `bonus/solutions/Lab_2_solution.ipynb` | 12 | 0 | 0 | No |
| `bonus/solutions/Lab_3_solution.ipynb` | 6 | 0 | 0 | No |
| `bonus/solutions/Lab_4_solution.ipynb` | 12 | 0 | 0 | No |
| `services_demo/1_Service.ipynb` | 10 | 0 | 0 | No |

The gaps are worth naming. In `1_Inference.ipynb` the three cells without an execution count are
the trailing curl comment, the `serve shutdown -y` cell, and an empty cell. In `2_Training.ipynb`
it is a single empty trailing cell. In `bonus/labs/Lab_4_RayTrain_AIAssisted.ipynb` two cells carry
an execution count from some earlier run but saved nothing, which is why it still counts as
unexecuted here.

Labs 1 and 3 were executed with `jupyter nbconvert --execute` on the laptop described under
[Tested configuration](#tested-configuration), and both finished with zero error outputs. Only the
setup cells and the vanilla PyTorch baseline in Lab 3 print anything. The exercises in both labs
are still `# TODO` stubs, so those cells now show an execution count and no result. Nothing was
solved for you.

## Model

`TwoTower` is a small embedding model:

- User and item embedding tables (default dim 64), L2-normalized.
- Training uses in-batch negatives: logits are a `[B, B]` matrix of user–item similarities; the diagonal is the positive pair.

`Recommend` loads a checkpoint with `torch.load(..., weights_only=True)` and returns top-k item ids per user.

## Planned compute

From `outline-compute.md`:

```yaml
compute_config:
  head_node:
    instance_type: m5.2xlarge
  worker_nodes:
    - instance_type: m5.2xlarge
      min_nodes: 2
      max_nodes: 2
```

The Anyscale demo config uses `head-2a10g-small:2` and gives the Recommender `num_gpus: 0.5`.

## A note on scope

This repo is course material, not a product, and most of it cannot run on an ordinary machine.
The honest limits:

**Two notebooks run locally, and they are the small ones.** `bonus/labs/Lab_1_RayServe_Fundamentals.ipynb`
and `bonus/labs/Lab_3_RayTrain_Fundamentals.ipynb` were executed on a CPU-only laptop and their
outputs are committed. Everything else in the repo was left as it came, because it needs a GPU,
an Anyscale service, or the course cluster paths under `/mnt`.

**The two main notebooks were not re-run here.** `1_Inference.ipynb` and `2_Training.ipynb` carry
outputs recorded on the course cluster, which the `/home/ray/anaconda3/lib/python3.11` paths in
those outputs confirm. They read `/mnt/cluster_storage/ecom/`, which does not exist off that
cluster, so the saved outputs are a record of someone else's run and not something reproduced here.

**The Anyscale pieces are closed to outsiders.** `services_demo/anyscale_serve_config.yaml` names
the image `anyscale/image/c26:5`, the compute config `head-2a10g-small:2`, and the cloud
`education-us-west-2`. All three are scoped to the course account, so `anyscale service deploy`
will not run as written from anywhere else. `serve_updated.yaml` is milder. It carries no image or
cloud, but its `working_dir` points at a zip in the `anyscale-materials` S3 bucket, which this repo
does not control either.

**The data is borrowed.** The sample catalog and user data come from the public bucket
`s3://anyscale-public-materials-use2/ecom`. Anyscale owns that bucket and can retire it whenever
it likes. Nothing here mirrors the data, so the notebooks that download it will stop working the
day it goes away.

**The pins are transcribed, not tested.** `requirements.txt` copies the pip pins out of the
`Dockerfile`. That set has never been resolved or installed in one go here, and the local
environment used for the labs already drifts from it.

**The solutions are unverified.** `bonus/solutions/` ships without outputs, and nobody executed
those notebooks in preparing this repo. Treat them as reference answers rather than as tested code.

**The slides describe the course, not this repo.** `offline_slides/` is the deck as delivered.
Where it and the code disagree, the code is what is actually here.
