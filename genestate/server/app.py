from contextlib import asynccontextmanager
from io import BytesIO
import logging
import time
import anyio  # Behoben: anyio ist nun sauber geladen
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from genestate.engine.sdxlengine import SDXLEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

engine = SDXLEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_model()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="GenEstate AI Engine", lifespan=lifespan)

class GenerateParams(BaseModel):
    style: str = "modern"
    size: str = "medium"
    roof_color: str = "dark"
    extra: str = "none"

HTML_INTERFACE = """
<!DOCTYPE html>
<html>
<head>
    <title>GenEstate AI Engine</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #0f172a; color: #fff; padding: 2rem; max-width: 800px; margin: auto; }
        .controls { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
        label { font-size: 0.85rem; color: #94a3b8; }
        select, button { width: 100%; padding: 0.75rem; background: #1e293b; color: #fff; border: 1px solid #334155; border-radius: 6px; font-size: 1rem; }
        button { grid-column: span 2; background: #2563eb; border: none; cursor: pointer; font-weight: bold; }
        button:disabled { background: #475569; cursor: not-allowed; }
        .preview { margin-top: 1rem; text-align: center; min-height: 400px; background: #1e293b; border-radius: 8px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
        img { max-width: 100%; height: auto; display: none; }
        #status { color: #94a3b8; font-size: 1rem; }
    </style>
</head>
<body>
    <h2>GenEstate AI - Performance Studio</h2>
    <div class="controls">
        <div>
            <label>Style</label>
            <select id="style">
                <option value="modern">Modern Villa</option>
                <option value="wooden">Wooden Cabin</option>
                <option value="mediterranean">Mediterranean</option>
            </select>
        </div>
        <div>
            <label>Size</label>
            <select id="size">
                <option value="small">Small</option>
                <option value="medium" selected>Medium</option>
                <option value="large">Large</option>
            </select>
        </div>
        <div>
            <label>Roof Color</label>
            <select id="roof_color">
                <option value="dark">Dark Slate</option>
                <option value="red">Red Tile</option>
                <option value="light">Beige Tile</option>
            </select>
        </div>
        <div>
            <label>Extra Feature</label>
            <select id="extra">
                <option value="none">None</option>
                <option value="pool">Pool</option>
                <option value="garage">Garage</option>
            </select>
        </div>
        <button id="btn" onclick="generate()">Generate Ultra-Fast</button>
    </div>
    <div class="preview">
        <span id="status">Select options and click Generate</span>
        <img id="result" alt="Generated Output" />
    </div>
    <script>
        async function generate() {
            const btn = document.getElementById('btn');
            const status = document.getElementById('status');
            const img = document.getElementById('result');
            btn.disabled = true;
            status.style.display = 'block';
            status.innerText = 'Running GPU Inference...';
            img.style.display = 'none';

            const payload = {
                style: document.getElementById('style').value,
                size: document.getElementById('size').value,
                roof_color: document.getElementById('roof_color').value,
                extra: document.getElementById('extra').value
            };

            try {
                const res = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error('Generation failed');
                const blob = await res.blob();
                img.src = URL.createObjectURL(blob);
                img.style.display = 'block';
                status.style.display = 'none';
            } catch (err) {
                status.innerText = 'Error: ' + err.message;
            } finally {
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_INTERFACE

@app.post("/generate")
async def generate_endpoint(data: GenerateParams):
    try:
        start_total = time.time()
        
        prompt = (
            f"eye level exterior shot, photograph of a single {data.size} {data.style} house, "
            f"{data.roof_color} roof, clear sky, neat grass lawn, architectural design"
        )
        if data.extra == "pool":
            prompt += ", a single small swimming pool in the backyard"
        elif data.extra == "garage":
            prompt += ", attached single-car garage on the side"

        negative_prompt = (
            "multiple pools, repeated pools, water everywhere, duplicate structures, "
            "aerial view, bird eye view, miniature, distortion, noise, blurry"
        )

        # Offload prediction task into thread pool to prevent blocking asyncio loop
        image, _ = await anyio.to_thread.run_sync(engine.generate, prompt, negative_prompt)
        
        img_io = BytesIO()
        image.save(img_io, format="WEBP", quality=80, method=0)
        img_io.seek(0)
        
        logging.info(f"TOTAL ENDPOINT RESPONSE TIME: {time.time() - start_total:.2f}s")
        return StreamingResponse(img_io, media_type="image/webp")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))