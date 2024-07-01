from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
import requests
import os

app = FastAPI()

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}


class GenerateImagesRequest(BaseModel):
    prompt: str
    styles: List[str]


@app.get("/health-check")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/generate-images")
async def generate_images(request: GenerateImagesRequest):
    prompt = request.prompt
    styles = request.styles
    print(prompt)
    images_paths = []
    for style in styles:
        image_bytes = query({"inputs": f"{prompt} style {style}"})

        # Define the filename for the current style
        filename = f"generated_image_{style}.jpg"

        # Check if the directory exists, create it if not
        os.makedirs("images", exist_ok=True)

        # Save the image to the local filesystem
        with open(os.path.join("images", filename), "wb") as img_file:
            img_file.write(image_bytes)

        images_paths.append(filename)

    return {"images": images_paths}


@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join("images", image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


def query(payload):
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
