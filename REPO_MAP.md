<!-- Aider Studio repo map — auto-generated. Do not edit by hand. -->
# Repository Map

_20 files indexed · 0 summarized. Summaries, outlines (with line numbers) and metadata only — NOT file contents. To work on a file's actual code, `/read` or `/add` it by the path shown._

## .gitignore
- __pycache__/

## benchmarks/__init__.py
- (empty)

## benchmarks/bench_inference.py
- (empty)

## cppcuda/readme.md
- here will cuda and c++ optimizations go

## genestate/__init__.py
- (empty)

## genestate/builder/__init__.py
- (empty)

## genestate/builder/buildermd.md
- .

## genestate/config.py
- class Config:  ·L4

## genestate/engine/__init__.py
- (empty)

## genestate/engine/base.py
- (empty)

## genestate/engine/sdxlengine.py
- class SDXLEngine:  ·L10
- def __init__(self):  ·L11
- def load_model(self):  ·L14
- def _warmup(self):  ·L34
- def generate(self, prompt: str, negative_prompt: str, seed: int = None):  ·L48

## genestate/server/__init__.py
- (empty)

## genestate/server/app.py
- async def lifespan(app: FastAPI):  ·L18
- class GenerateParams(BaseModel):  ·L26
- def index():  ·L130
- async def generate_endpoint(data: GenerateParams):  ·L134

## genestate/server/router.py
- (empty)

## kernels/__init__.py
- print("not implemented")

## kernels/kernelsmd.md
- here ill put triton for example

## README.md
- high performance generative ai inference project

## requirements.txt
- torch

## scripts/run_server.py
- import uvicorn

## tests/testengine.py
- (empty)
