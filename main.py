import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi.responses import FileResponse
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Hugging Face API URLs and headers
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
IMAGE_GENERATION_URL = "https://api-inference.huggingface.co/models/mann-e/Mann-E_Dreams"
CAPTION_GENERATION_URL = 'https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct/v1/chat/completions'
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    styles: List[str] = []


@app.get("/health-check")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = request.prompt
    styles = request.styles
    images_urls = []
    captions = []

    for style in styles:
        # Generate image
        image_bytes = query_image({"inputs": f"{prompt} style {style}"})

        # Encode image in base64
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare payload for image upload API
        upload_payload = {
            "key": os.getenv('IMAGE_KEY'),
            "action": "upload",
            "source": encoded_image,
            "format": "json"
        }

        # Upload image to hosting service
        upload_response = requests.post(
            "https://freeimage.host/api/1/upload", data=upload_payload)
        upload_data = upload_response.json()

        # Extract the URL from the response
        image_url = upload_data.get('image', {}).get('url')

        if image_url:
            images_urls.append(image_url)

            # Generate caption for the current image
            caption_payload = {
                "model": "meta-llama/Meta-Llama-3-70B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Generate a unique social media post short caption with tags for the image at {image_url}. The image was generated based on the prompt '{prompt}' with a focus on the '{style}' style. Please include hashtags relevant to the image content.Give only the caption and no extra text or quotation marks"
                    }
                ],
                "max_tokens": 500,
                "stream": False
            }
            caption_response = requests.post(
                CAPTION_GENERATION_URL, headers=headers, json=caption_payload)
            data = caption_response.json()
            generated_caption = data["choices"][0]["message"]["content"].strip(
            )
            captions.append(generated_caption)

    # Return image URLs and their corresponding captions
    return {"images": images_urls, "captions": captions}


@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join("images", image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


def query_image(payload):
    response = requests.post(IMAGE_GENERATION_URL,
                             headers=headers, json=payload)
    response.raise_for_status()
    return response.content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))
