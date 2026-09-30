# Sci-Vizio web app: architecture and deployment

This guide takes the multi-user web app (the `sci_vizio_saas` backend and the `frontend/` React app) from a fresh clone to production. It covers:

* Firebase Hosting for the frontend and Firebase Authentication for sign-in;
* Heroku for the API, the worker and PostgreSQL;
* S3-compatible object storage for files;
* Stripe for billing.

The command-line pipeline in the main [README](../README.md) is unaffected and still works on its own.

---

## 1. Architecture

```
 Browser (React SPA on Firebase Hosting)
   │  Firebase JS SDK ──────────────► Firebase Authentication (email/password, Google)
   │  Authorization: Bearer <ID token>
   ▼
 Heroku "web" dyno: FastAPI (sci_vizio_saas.main)
   │  verifies ID tokens (firebase-admin)             ┌──────────────────────────────┐
   │  issues signed upload/download URLs ───────────► │ Object storage (R2 / S3)     │
   │                                                  │  users/<uid>/documents/<id>/ │
   ▼                                                  │   original.pdf, figures/*.png│
 Heroku Postgres (pgvector)                           └──────────────▲───────────────┘
   users, collections, documents, figures (+ embeddings),            │
   jobs (queue), stripe_events                                       │
   ▲                                                                 │
 Heroku "worker" dyno: python -m sci_vizio_saas.worker ──────────────┘
   claims jobs (SELECT … FOR UPDATE SKIP LOCKED), extracts figures,
   calls the vision model (OpenRouter), embeds text (fastembed), stores figures

 Stripe Checkout / Customer Portal ──webhooks──► /api/billing/webhook
```

**Main design decisions**

| Decision | Why |
|---|---|
| PostgreSQL for *all* app data, including the job queue and vectors | One database to back up and reason about. pgvector is available on every Heroku Postgres plan, and `SKIP LOCKED` makes a reliable queue without Redis. |
| Browsers upload and download directly to object storage via signed URLs | PDF bytes never pass through dynos, which avoids Heroku's 30-second request limit and saves memory. The API checks size and type after upload, and the worker verifies the SHA-256 checksum. |
| Exact vector search, filtered by user | Each user has at most a few thousand figures. Scanning them exactly is fast, and it never drops results the way an approximate index combined with a per-user filter can. |
| fastembed (ONNX) instead of torch | The image stays small and fits a 512 MB Basic dyno. Its vectors are identical to the CLI's sentence-transformers embeddings (this is tested). |
| Docker image deployed via `heroku.yml` | The root `requirements.txt` includes torch and chromadb for the CLI. The container installs only `requirements-saas.txt` and bakes in the embedding model. The same image also runs anywhere else. |

---

## 2. Object storage: recommendation

**Use Cloudflare R2.** The alternatives are covered by the same code (`STORAGE_BACKEND=s3` works with any S3-compatible API):

| | Cloudflare R2 (recommended) | AWS S3 | Google Cloud Storage / Firebase Storage |
|---|---|---|---|
| Egress (users viewing figures and downloading PDFs) | **Free** | ~$0.09/GB | ~$0.12/GB |
| Storage | $0.015/GB-month, first 10 GB free | ~$0.023/GB-month | ~$0.020–0.026/GB-month |
| Works with this app | S3 API with signed PUT/GET | Native | S3 interoperability mode with HMAC keys |
| Notes | No presigned *POST*, so the upload size is checked after upload (implemented) | Most mature; pick it if you're already on AWS | Same Google account as Firebase, but egress costs grow with usage |

This product is read-heavy: every search result loads figure images. With R2's free egress, storage cost stays roughly proportional to the number of PDFs rather than to traffic. Prices are approximate as of 2026; check each provider's current pricing.

---

## 3. Freemium model

The limits and price come from environment variables. These values are placeholders:

| | Free | Premium |
|---|---|---|
| Price | $0 | **$9 / month** (`PREMIUM_PRICE_MONTHLY_USD`, charged via `STRIPE_PRICE_ID`) |
| PDFs | 10 (`FREE_MAX_DOCUMENTS`) | 500 (`PREMIUM_MAX_DOCUMENTS`) |
| Max file size | 20 MB | 100 MB |
| Figures analysed per PDF | 40 | 200 |

**Rules implemented**

* Every stored PDF counts towards the quota, including uploads in progress. The check locks the user's row, so parallel uploads can't exceed the limit.
* Uploads that are never completed are removed after 2 hours by the worker's maintenance loop.
* Premium is active while the Stripe subscription is `active`, `trialing` or `past_due`. `past_due` covers Stripe's automatic payment retries. After cancellation the user drops to Free: existing documents stay searchable, but new uploads are blocked until they are under the Free limit.
* Duplicate PDFs (same SHA-256) are rejected per user and don't count twice.
* To keep costs down, the worker skips images smaller than 100 px (`MIN_FIGURE_SIDE`) and images repeated across pages, *before* any vision-model call. On a sample of 346 real papers, about 45% of extracted images were decorative (icons, logos, gradients). These are still indexed but hidden from search by default.

**Unit economics.** These are rough numbers; verify them against your chosen model's price.

* **Cost driver:** one vision-model call per figure. A cheap "flash" model on OpenRouter costs roughly $0.0002–$0.001 per image, which puts a figure-heavy paper at about $0.01–$0.04.
* **Free tier:** a free user who fills their quota costs about $0.10–$0.40 in total. That is a one-off cost, because figures are processed once.
* **Premium tier:** a Premium user who uploads 500 figure-heavy PDFs in one month could cost $5–$20 of vision calls.
  * **Mitigation:** add a monthly processing allowance (e.g. N new PDFs per month), or pick a cheaper `VISION_MODEL`.
  * Revisit this before launch.

---

## 4. Prerequisites

* Accounts: Firebase (Google), Heroku, Cloudflare (or AWS), Stripe, OpenRouter.
* CLIs: `firebase` (firebase-tools), `heroku`, and optionally `stripe` for testing webhooks locally.
* Node 20+ and Python 3.10+ for local development.

> **Heroku platform status.** On 6 February 2026 Heroku announced a move to a "sustaining engineering" model: continued stability, security and support, and no new Enterprise contracts. Pay-as-you-go accounts are unaffected for now. The backend is a standard Docker image configured entirely by environment variables. If you ever need to move, it runs unchanged on Fly.io, Render, Railway or Google Cloud Run with any PostgreSQL that has pgvector (Neon, Supabase, Cloud SQL, RDS).

---

## 5. Firebase: project and authentication

1. Create a project in the [Firebase console](https://console.firebase.google.com). Note its **project ID**.
2. Go to **Authentication > Sign-in method** and enable **Email/Password** and **Google**.
3. Go to **Authentication > Settings > Authorized domains**. `<project>.web.app` and `<project>.firebaseapp.com` are there by default. Add your custom domain if you use one.
4. Go to **Project settings > General > Your apps** and add a **Web app**. Copy `apiKey`, `authDomain`, `projectId` and `appId` for the frontend (section 9).
5. Go to **Project settings > Service accounts** and generate a new private key (a JSON file). Its contents become `FIREBASE_CREDENTIALS_JSON` on Heroku.
   * This key is only used to delete a user's Firebase account when they delete their Sci-Vizio account. Token verification doesn't need it.
   * Keep this file secret; never commit it.
6. Point the repo at your project:
   ```bash
   firebase use --add        # choose the project; updates .firebaserc
   ```

Optional: to stop people creating many free accounts with throwaway addresses, set `REQUIRE_VERIFIED_EMAIL=true`. Then add a "verify your email" step to sign-up (`sendEmailVerification`).

---

## 6. Object storage: Cloudflare R2

1. Go to **R2 > Create bucket**, for example `sci-vizio-prod`. Keep it **private**; don't enable public access.
2. Go to **R2 > Manage API tokens > Create API token** with *Object Read & Write* on that bucket. Note the **Access Key ID**, the **Secret Access Key** and the **S3 endpoint** (`https://<account-id>.r2.cloudflarestorage.com`).
3. Go to **Bucket > Settings > CORS policy**. Browsers upload and download directly, so they need CORS:
   ```json
   [
     {
       "AllowedOrigins": ["https://<project>.web.app", "https://<project>.firebaseapp.com"],
       "AllowedMethods": ["GET", "PUT"],
       "AllowedHeaders": ["Content-Type"],
       "MaxAgeSeconds": 3600
     }
   ]
   ```
4. Recommended: add a lifecycle rule that deletes incomplete multipart uploads after 1 day.

---

## 7. Backend on Heroku (API, worker, Postgres)

```bash
heroku create sci-vizio-api                  # or your own name
heroku stack:set container                   # build from Dockerfile via heroku.yml
heroku addons:create heroku-postgresql:essential-0   # sets DATABASE_URL; pgvector is supported

heroku config:set \
  ENVIRONMENT=production \
  FIREBASE_PROJECT_ID=<project-id> \
  FIREBASE_CREDENTIALS_JSON="$(cat path/to/service-account.json)" \
  STORAGE_BACKEND=s3 \
  S3_BUCKET=sci-vizio-prod \
  S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com \
  S3_ACCESS_KEY_ID=<r2-key-id> \
  S3_SECRET_ACCESS_KEY=<r2-secret> \
  OPENROUTER_API_KEY=<sk-or-...> \
  VISION_MODEL=qwen/qwen3.5-flash-02-23 \
  FRONTEND_URL=https://<project>.web.app \
  CORS_ORIGINS=https://<project>.web.app,https://<project>.firebaseapp.com \
  PUBLIC_API_URL=https://sci-vizio-api-<hash>.herokuapp.com

git push heroku main                         # builds the image, runs migrations, releases
heroku ps:scale web=1:basic worker=1:basic
heroku open /api/health                      # → {"status":"ok"}
```

* **Migrations** run automatically in Heroku's *release phase* (`alembic upgrade head`, see `heroku.yml`). A failed migration aborts the release, and the old version keeps serving.
* **Startup check:** with `ENVIRONMENT=production` the app refuses to start without Firebase, S3 storage and an OpenRouter key. This catches configuration mistakes at deploy time.
* **Connections:** Essential-0 allows 20 connections. The API and worker use small pools (5 and 2).
* **Scaling:** to process more PDFs in parallel, add worker dynos (`worker=2`). Jobs are claimed with row locks, so workers never process the same PDF twice. `FIGURE_CONCURRENCY` (default 4) sets parallel vision calls per worker.
* **Backups:** set a schedule with `heroku pg:backups:schedule DATABASE_URL --at '02:00 Europe/London'`. Object storage has no built-in versioning, so turn on bucket versioning or replication if you need it.
* **Rough monthly platform cost:** about $19 plus usage. That is 2 Basic dynos at $7 each and Postgres Essential-0 at $5; storage is free up to 10 GB on R2; Firebase Auth and Hosting are free at small scale.

---

## 8. Stripe billing

1. In the Stripe dashboard (start in **test mode**), create a **Product** called "Sci-Vizio Premium" with a **recurring price** of $9/month. Note the **price ID** (`price_…`).
2. Go to **Settings > Billing > Customer portal** and enable it. Allow cancelling, updating payment methods and viewing invoices.
3. Go to **Developers > Webhooks > Add endpoint**:
   * URL: `https://<your-heroku-app>/api/billing/webhook`
   * Events: `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated`, `customer.subscription.deleted`
   * Note the **signing secret** (`whsec_…`).
4. Configure the API:
   ```bash
   heroku config:set STRIPE_SECRET_KEY=sk_test_... STRIPE_PRICE_ID=price_... STRIPE_WEBHOOK_SECRET=whsec_... \
     PREMIUM_PRICE_MONTHLY_USD=9
   ```
   Without these variables the app runs with billing disabled: the upgrade button is hidden and checkout returns 503.
5. Test end to end with card `4242 4242 4242 4242`, then switch to live keys and a live webhook.

The webhook handler verifies signatures, ignores redelivered events (`stripe_events` table), and derives the plan only from Stripe's subscription status. Keep `PREMIUM_PRICE_MONTHLY_USD` in sync with the Stripe price: it is only what the UI displays.

For local testing: `stripe listen --forward-to localhost:8000/api/billing/webhook`.

---

## 9. Frontend on Firebase Hosting

```bash
cd frontend
cp .env.example .env.production
# Fill in: VITE_API_BASE_URL=https://<your-heroku-app>
#          VITE_FIREBASE_API_KEY / AUTH_DOMAIN / PROJECT_ID / APP_ID from section 5
cd ..
firebase deploy --only hosting               # predeploy runs `npm ci && npm run build`
```

The site is served from `https://<project>.web.app`. `firebase.json` configures:

* single-page-app rewrites;
* long-lived caching for hashed assets;
* basic security headers.

Firebase web config values are not secrets; they identify the project. Access is enforced by the API, which verifies every ID token.

---

## 10. Deploying from GitHub Actions (optional)

`.github/workflows/deploy.yml` is a **manual** workflow: open the Actions tab, choose *deploy* and click *Run workflow*. It pushes the backend to Heroku, then builds and deploys the frontend. Configure it under **Settings > Secrets and variables > Actions**:

| Kind | Name | Value |
|---|---|---|
| Secret | `HEROKU_API_KEY` | `heroku authorizations:create` |
| Secret | `HEROKU_APP` | e.g. `sci-vizio-api` |
| Secret | `FIREBASE_SERVICE_ACCOUNT` | Service-account JSON with the *Firebase Hosting Admin* role |
| Variable | `VITE_API_BASE_URL`, `VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, `VITE_FIREBASE_PROJECT_ID`, `VITE_FIREBASE_APP_ID` | As in section 9 |

`tests.yml` runs on every push and pull request. It runs the Python tests (including PostgreSQL + pgvector) and the frontend typecheck, tests and build.

---

## 11. Local development

```bash
# 1. Database
docker compose up -d                                  # Postgres + pgvector on :5432
#    (or any local Postgres with pgvector)

# 2. Backend
pip install -e ".[saas,dev]"
export DATABASE_URL=postgresql://sci:sci@localhost:5432/sci_vizio
export FIREBASE_PROJECT_ID=demo-sci-vizio FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099
export OPENROUTER_API_KEY=sk-or-...                   # real vision calls (cheap models: cents per paper)
alembic -c sci_vizio_saas/alembic.ini upgrade head
uvicorn sci_vizio_saas.main:create_app --factory --reload --port 8000
python -m sci_vizio_saas.worker                       # in another terminal

# 3. Auth emulator (no Firebase project needed)
firebase emulators:start --only auth --project demo-sci-vizio   # UI at http://localhost:4000

# 4. Frontend
cd frontend && cp .env.example .env.local             # set VITE_USE_AUTH_EMULATOR=true
npm install && npm run dev                            # http://localhost:5173
```

In development, files go to `var/storage/` through the API's signed `/api/files` route (`STORAGE_BACKEND=local`). API docs are at <http://localhost:8000/api/docs>.

**Tests**

```bash
SCI_TEST_POSTGRES_DSN=postgresql://sci:sci@localhost:5432/sci_vizio pytest   # backend + library
cd frontend && npm test                                                      # frontend
SCI_TEST_REAL_MODELS=1 pytest tests/test_embeddings.py                        # optional: real models
```

---

## 12. API overview

All routes are under `/api`, and all except `health`, `plans`, `billing/webhook` and the signed `files` URLs require `Authorization: Bearer <Firebase ID token>`.

| Method & path | Purpose |
|---|---|
| `GET /plans` | Public plan list and prices |
| `GET/PATCH/DELETE /me` | Profile, plan, usage; delete account (cancels subscription, deletes files, rows and Firebase user) |
| `GET/POST /collections`, `PATCH/DELETE /collections/{id}` | Organise PDFs; deleting a collection keeps its PDFs |
| `GET /documents?collection=&status=&q=` | List documents (`collection=unfiled` for documents without one) |
| `POST /documents/uploads` → `PUT <signed url>` → `POST /documents/{id}/complete` | Upload flow |
| `GET/PATCH/DELETE /documents/{id}`, `GET /documents/{id}/download`, `POST /documents/{id}/retry`, `GET /documents/{id}/figures` | Manage one document |
| `POST /search` | Semantic search with `figure_types`, `collections`, `documents` filters and facet counts |
| `POST /billing/checkout`, `POST /billing/portal`, `POST /billing/webhook` | Stripe |

The full OpenAPI schema is served at `/api/docs`.

---

## 13. Known limitations and next steps

* **Vector-drawn figures** (common in LaTeX papers) are not extracted, because only embedded raster images are. See recommendation 3 in [STATE_OF_THE_ART.md](STATE_OF_THE_ART.md): layout-aware figure detection and rendering.
* **No rate limiting** on the API yet. Add it at the edge (e.g. Cloudflare in front of Heroku) or with a middleware before a public launch.
* **No monthly processing allowance for Premium** (see the unit economics in section 3).
* **Emails** (verification, receipts) rely on Firebase and Stripe defaults. There are no product emails such as "your PDF is ready".
* **Legal and compliance:** terms of service, a privacy policy, a DPA with the vision-model provider (uploaded papers are sent to it), and GDPR export. Account deletion is implemented; data export is not.
