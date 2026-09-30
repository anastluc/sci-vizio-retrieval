# How relevant is Sci-Vizio-Retrieval today?

*A landscape review against current academic and industry systems — September 2026*

---

## 1. Summary

Sci-Vizio-Retrieval uses a **verbalize-then-embed** design. It pulls images out of PDFs, has a vision LLM write a structured JSON description of each one, embeds that text, and runs semantic search over it. The design is not out of date. In 2025–2026 two things happened:

* **Industry adopted the same pattern.** Azure AI Search ("image verbalization"), Amazon Bedrock Knowledge Bases (Bedrock Data Automation) and Google's Vertex AI RAG Engine with Gemini parsing all turn images into LLM-written text and index that text. They now offer it as a managed, configurable feature.
* **Recent research supports text for scientific papers.** ArXivDoc (Khalighinejad et al., 2026) found that text-based representations work best on scientific papers, *even for figure-retrieval queries*, and that page-as-image retrievers degrade as documents get longer. UniDoc-Bench (Salesforce, 2025) found that systems which **fuse** text and image evidence beat both text-only and joint-multimodal-embedding systems.

So the core idea is still sound. Its value today is **as a transparent, self-hostable, figure-level research tool and baseline**, not as a competitive product. The gaps are in everything around the core idea:

| Gap | Why it matters now |
|---|---|
| **Raster-only figure extraction** | 54 of 346 papers in this repo's own corpus (16%) produced **zero** images. Vector-drawn plots and diagrams, such as those in the Sentence-BERT paper (1908.10084), are invisible to `page.get_images()`. Many other "images" are fragments (18 per paper on average). |
| **No captions or surrounding context** | The best-performing approaches index the figure *with* its caption and referencing paragraphs. This pipeline indexes the LLM's view of the pixels only. |
| **No evaluation** | There are no relevance judgments or metrics, so no claim about retrieval quality can be checked. Public benchmarks exist that it could use (SciMMIR, CharXiv, ArXivDoc, SciFIBench). |
| **Single dense retriever** | There is no lexical/hybrid search, no reranker and no multi-vector (late-interaction) option. All of these are standard in 2026. |

**Recommendation:** keep the project and position it as an open, inspectable **scientific-figure search baseline**. Close the four gaps above; section 6 lists them in order of payoff per effort. The new PostgreSQL backend helps here: hybrid search, metadata filtering and multi-user deployment become incremental SQL work instead of a new infrastructure layer.

---

## 2. What the system does

| Stage | Implementation | 2026 alternatives |
|---|---|---|
| PDF → figures | PyMuPDF, embedded raster images per page | Layout-aware parsers: Docling, MinerU, Marker, GROBID/PDFFigures 2.0; or render whole pages |
| Figure → representation | Vision LLM via OpenRouter → JSON (`image_type`, `title`, `description`, axes, labels…) | Same (verbalization); multimodal embeddings (Cohere Embed v4, Jina v4, Voyage multimodal 3.5, Nova MME); late-interaction (ColPali/ColQwen, Nemotron ColEmbed) |
| Representation → index | MiniLM text embedding in ChromaDB **or PostgreSQL + pgvector (new)**; CLIP image embeddings in Chroma | Hybrid BM25 + dense, multi-vector stores (Vespa, Qdrant, Weaviate, Milvus), rerankers |
| Query UI | Gradio, text → top-k figures + JSON | Chat/QA with citations (Asta Paper+Figure QA, Elicit, SciSpace), agentic retrieval |

Its unit of retrieval is **the individual figure, together with a structured, human-readable description**. That is still somewhat unusual. Most current systems retrieve *pages* (the ColPali family) or *chunks* (text RAG), and treat figures as a side effect.

---

## 3. Academic landscape

### 3.1 Page-as-image retrieval (the "OCR-free" wave)

* **ColPali** (Faysse et al., ICLR 2025) embeds page screenshots with a VLM and scores them by ColBERT-style late interaction. It introduced the **ViDoRe** benchmark. The same authors followed with ColQwen2/2.5.
* **VisRAG** (Yu et al., ICLR 2025) and **M3DocRAG** (Cho et al., 2024) retrieve page images directly for a multimodal generator. VisRAG reports a 20–40% end-to-end gain over text-based RAG on its benchmarks.
* **Current leader:** NVIDIA's **Nemotron ColEmbed V2** (Feb 2026) ranked first on ViDoRe V3 (nDCG@10 ≈ 63.4) at 8B parameters. The Visual RAG Toolkit (2026) addresses the storage cost of multi-vector indexes with training-free pooling.
* **Survey:** Yan et al. (2026), *Unlocking Multimodal Document Intelligence*, frames the field as a representation-vs-efficiency trade-off for late interaction. It also notes a shift toward reasoning-intensive evaluation.

**Relevance to this project:** these systems are strongest on visually rich business documents (slides, reports, infographics). They are expensive at index time (GPU, many vectors per page) and retrieve pages, not figures.

### 3.2 Evidence that text still wins for scientific papers

* **ArXivDoc — *Document-as-Image Representations Fall Short for Scientific Retrieval*** (Khalighinejad et al., 2026) builds a benchmark from LaTeX sources. Its findings:
  1. Image-based document representations underperform, and get worse as document length grows.
  2. **Text representations are most effective overall, even for figure queries.**
  3. Interleaved text + image beats image-only without extra training.
* **UniDoc-Bench** (Peng et al., Salesforce, 2025) compares text-only, image-only, joint multimodal embedding and **text-image fusion** RAG. Fusion consistently wins.

**Relevance:** this is the strongest argument that the project's architecture is appropriate. Describing figures with text is the right default for scientific literature. What it lacks is the *text around the figure* (captions, the paragraphs that cite it) and a fusion step.

### 3.3 Scientific figure understanding and retrieval benchmarks

| Benchmark | What it measures | Use for this project |
|---|---|---|
| **SciMMIR** (ACL Findings 2024) | Figure/table ↔ caption retrieval across arXiv | Direct retrieval eval of the index |
| **CharXiv** (NeurIPS 2024) | 2,323 arXiv charts, 10K+ descriptive/reasoning questions | Quality of the VLM descriptions |
| **SciFIBench** (2024) | Figure interpretation by LMMs | Choosing the vision model |
| **ArXivDoc** (2026) | Scientific retrieval incl. figure queries | End-to-end retrieval eval |
| **S1-MMAlign** (2026) | Large multi-disciplinary figure–text alignment | Training/eval data for fusion |
| **Diagram-MMU**, **SciFigQual-Bench** (2026) | Diagram understanding; figure quality | Secondary |
| **MMDocIR**, **UniDoc-Bench** (2025) | Long-document multimodal retrieval / RAG | Page- vs figure-level comparison |

The field has good, *figure-specific* evaluation data. Adopting one of these benchmarks is the cheapest way to turn this repository from a demo into a measurable system.

### 3.4 Document parsing

* **MinerU** (OpenDataLab) and **Docling** (IBM) do layout-aware parsing. They detect figure regions (including vector graphics), tables and formulas, and export figures as images. MinerU is particularly strong on multi-column scientific layouts and LaTeX formulas, though recent work reports imperfect figure–caption pairing.
* Older but still-used scientific tools: **GROBID** (structure and references) and **PDFFigures 2.0 / DeepFigures** (figure and caption extraction).

**Relevance:** replacing raw `get_images()` with one of these would fix the largest measured weakness (vector figures, fragments) *and* supply captions.

---

## 4. Industry landscape

### 4.1 Managed cloud "multimodal RAG"

| Offering | Approach | Overlap with this project |
|---|---|---|
| **Azure AI Search** multimodal | *Image verbalization* via a GenAI Prompt skill, **or** direct image embeddings; cropped images kept in a knowledge store | Almost the same pipeline, managed |
| **Amazon Bedrock Knowledge Bases** multimodal retrieval (GA Nov 2025) | Nova Multimodal Embeddings (native) **or** Bedrock Data Automation (convert to text) | Same two options as this repo (CLIP vs. verbalization) |
| **Google Vertex AI RAG Engine** + Document AI / Gemini | Layout parsing + Gemini multimodal understanding | Same idea, with better parsing |

**Takeaway:** the pattern is now a commodity in enterprise clouds. This project cannot compete with them as general infrastructure. It stands apart in being **open, cheap (bring your own OpenRouter model), portable, and specialised for scientific figures with a structured schema** (axes, labels, time periods, key patterns).

### 4.2 Embedding models

* **Cohere Embed v4**: text, images and mixed PDFs in one space, 128K context. Reported as the top performer on MTEB multimodal tasks as of Q2 2026.
* **Jina Embeddings v4**: open weights on Qwen2.5-VL-3B, with task LoRA adapters and single- or multi-vector output.
* **Voyage multimodal 3.5** (API-only), **SigLIP-2**, **JinaCLIP-v2**.
* The CLIP collection this project writes into Chroma (OpenCLIP, generic web-image training) is not used by the UI, and it is weak on charts and diagrams. A document-tuned model (Jina v4, a ColQwen variant or Nemotron ColEmbed) would be the modern replacement if image-side retrieval is wanted.

### 4.3 Vector storage

* **pgvector** 0.8.x (current 0.8.2, Feb 2026) adds *iterative index scans*, so filtered ANN queries (e.g. `WHERE pdf_file = …`) return complete top-k results. **pgvectorscale** adds StreamingDiskANN for larger corpora. Postgres full-text search allows **hybrid search without a sidecar**, and managed Postgres everywhere ships pgvector.
* **ChromaDB** (1.x, Rust core) remains the easiest embedded option for local, single-user work.
* **Vespa, Qdrant, Weaviate and Milvus** natively support multi-vector/late-interaction (ColPali-style) search, which pgvector does not. They are the right choice only if page-level ColPali retrieval is adopted.

The PostgreSQL backend added in this release is a good fit for where the project should go next. Hybrid lexical+dense search, metadata filters (year, venue, figure type), and joins with bibliographic data (e.g. Semantic Scholar/OpenAlex metadata) all become SQL rather than new services.

### 4.4 End-user research products

* **Ai2 Asta** (launched Aug 2025) includes **Paper + Figure QA** over 12M+ full-text papers. It runs on the Semantic Scholar corpus (200M+ papers), which also feeds Elicit, SciSpace, Consensus and Undermind.
* These products answer *questions* with citations. None exposes a transparent **figure search engine** you can run on *your own* PDF collection (a private lab library, internal reports, or papers under embargo).

**Takeaway:** figure-level search over a private corpus, with inspectable structured descriptions, is still an underserved niche. It is where this project remains relevant.

---

## 5. Positioning

| Dimension | Sci-Vizio-Retrieval | ColPali-family | Cloud multimodal RAG | Asta / Elicit etc. |
|---|---|---|---|---|
| Unit of retrieval | **Figure** | Page | Chunk / image | Paper / answer |
| Representation | Structured LLM description (JSON) | Multi-vector page embedding | Text or native embedding | Text |
| Corpus | **Your PDFs** | Your docs | Your docs | Public literature |
| Self-hostable / open | **Yes** | Yes (GPU) | No | No |
| Index-time cost | 1 VLM call per image (API) | GPU per page | Metered | — |
| Interpretability | **High** (human-readable JSON) | Low | Medium | Medium |
| Measured quality | **None yet** | ViDoRe | Vendor | Vendor |
| Vector-graphics figures | **Missed** | Captured (page render) | Depends on parser | — |

---

## 6. Recommendations (ordered by payoff / effort)

1. **Add an evaluation harness.** Take a SciMMIR or ArXivDoc subset that overlaps the corpus, and report Recall@k / nDCG@10 per backend and per vision model. *(Low effort. It makes every later change measurable.)*
2. **Index captions and citing paragraphs with each figure.** The text is already extracted to `output/extracted_text/`. Concatenate it into `IndexRecord.document_text`. Both ArXivDoc and UniDoc-Bench suggest this is the largest single quality gain. *(Low effort.)*
3. **Replace raster extraction with layout-aware figure detection** (Docling or MinerU). Alternatively, render detected figure regions with PyMuPDF's `get_pixmap(clip=…)`. This recovers the 16% of papers that currently yield nothing and removes fragments. *(Medium effort.)*
4. **Hybrid search on Postgres.** Add a generated `tsvector` column plus reciprocal-rank fusion with the pgvector score, then an optional cross-encoder reranker. *(Medium effort; natural with the new backend.)*
5. **Swap CLIP for a document-tuned multimodal embedder** (Jina v4 / ColQwen) as a second signal, and fuse it with the text score. Otherwise, drop the unused CLIP collection to save index time. *(Medium effort.)*
6. **Structured-output mode for the vision call** (JSON schema / `response_format`) to eliminate the "extract the outermost braces" parsing. *(Low effort.)*

---

## 7. Verdict

**Relevant as a focused, open research tool and baseline; not competitive as a general document-RAG system.** Its central bet has become mainstream and has recent empirical support for scientific literature: verbalize figures with a VLM, then search the text. Its differentiators (figure-level granularity, structured and inspectable descriptions, private corpora, a cheap model-agnostic API, and now a production-grade Postgres backend) fill a real gap between enterprise multimodal RAG and public literature assistants. To stay credible it needs measured retrieval quality and modern figure extraction; recommendations 1–3 get it there with modest effort.

---

## Sources

* Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models* — https://arxiv.org/abs/2407.01449 · ViDoRe benchmark: https://github.com/illuin-tech/vidore-benchmark
* NVIDIA, *Nemotron ColEmbed V2* (2026) — https://arxiv.org/abs/2602.03992
* *Visual RAG Toolkit: Scaling Multi-Vector Visual Retrieval* (2026) — https://arxiv.org/abs/2602.12510
* Yu et al., *VisRAG* (ICLR 2025) — https://proceedings.iclr.cc/paper_files/paper/2025/file/3640a1997a4c9571cea9db2c82e1fc35-Paper-Conference.pdf
* Cho et al., *M3DocRAG* — https://huggingface.co/papers/2411.04952
* Yan et al., *Unlocking Multimodal Document Intelligence* (survey, 2026) — https://arxiv.org/abs/2602.19961
* Khalighinejad et al., *Document-as-Image Representations Fall Short for Scientific Retrieval* (2026) — https://arxiv.org/abs/2604.18508
* Peng et al., *UniDoc-Bench* — https://arxiv.org/abs/2510.03663
* *MMDocIR* — https://arxiv.org/abs/2501.08828
* *SciFIBench* — https://arxiv.org/abs/2405.08807
* *S1-MMAlign* (2026) — https://arxiv.org/abs/2601.00264
* *Diagram-MMU* (2026) — https://arxiv.org/abs/2608.12262 · *SciFigQual-Bench* (2026) — https://arxiv.org/abs/2607.27084
* *Awesome Multimodal RAG* list — https://github.com/JarvisUSTC/Awesome-Multimodal-RAG
* MinerU — https://arxiv.org/abs/2409.18839 · Docling/Marker/MinerU comparison — https://www.spheron.network/blog/self-host-document-intelligence-docling-marker-mineru-rag-guide/
* Azure AI Search multimodal / image verbalization — https://learn.microsoft.com/en-us/azure/search/multimodal-search-overview
* Amazon Bedrock Knowledge Bases multimodal retrieval — https://aws.amazon.com/about-aws/whats-new/2025/11/multimodal-retrieval-bedrock-knowledge-bases/
* Vertex AI multimodal RAG — https://medium.com/google-cloud/building-an-enterprise-grade-multimodal-rag-platform-on-google-vertex-ai-a-zero-to-production-e3e4b7c53056
* Multimodal embeddings (SigLIP-2, JinaCLIP-v2, Cohere Embed v4) — https://www.spheron.network/blog/multimodal-embedding-models-gpu-cloud-siglip2-jinaclip-cohere/ · https://milvus.io/blog/choose-embedding-model-rag-2026.md
* pgvector — https://github.com/pgvector/pgvector · 0.8.2 release — https://www.postgresql.org/about/news/pgvector-082-released-3245 · pgvectorscale & hybrid search — https://thebuild.com/blog/on-pgvectorscale-and-hybrid-search-without-an-elasticsearch-sidecar/
* Ai2 Asta — https://asta.allen.ai/ · Semantic Scholar — https://en.wikipedia.org/wiki/Semantic_Scholar
* In-repo measurement: `pdf_processing` table of this project's `pdf_processing.db` (346 non-duplicate PDFs; 54 with zero extracted images; mean 18.3 images per PDF).
