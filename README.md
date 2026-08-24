# Real-Time Search & Recommendation Systems for AI Commerce

Training materials for **Ray Summit 2026**. This course walks through batch inference, online serving, and periodic model updates for an e-commerce recommender plus semantic search stack built on Ray.

You start from a pre-trained two-tower recommender and a product catalog with description embeddings, then:

1. Run **batch recommendation** over a user dataset with Ray Data.
2. Serve **live recommendations and search** with composed Ray Serve deployments.
3. **Fine-tune** the recommender on fresh interaction data with Ray Train.
4. Optionally productionize the service on **Anyscale**.

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

### Serve from the CLI

```bash
# generate or inspect a config
serve build search_and_recommend:bound_ingress -o serve_config.yaml

# deploy
serve deploy serve_config.yaml
```

`serve_updated.yaml` is a thicker config: 4 DatabaseFacade replicas, 2 Recommender replicas, and autoscaling Search/Ingress. It also sets a remote `working_dir` zip so workers do not depend on your workspace files.

Query a running service (Ingress parses the body as a JSON *string*, same as the notebooks):

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
