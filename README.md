# What is this about

viziometrics - run vision LLM to describe an image → store → retrieve with semantic search

## Pipeline steps
```
python pdf_image_extractor.py
python image_processor.py
python image_processor_retry.py
python indexer.py
python gradio_app.py
```
### 1 Pdf extract to images

using minerU https://github.com/opendatalab/MinerU



### 2 Describe images and index to ES

### 3 Semantic search


## Setup environment

```
pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
```

