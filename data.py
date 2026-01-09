import json
from dotenv import load_dotenv
import requests
import os

import base64
import io
from pymilvus import MilvusClient, DataType

import os
from pathlib import Path

import cohere
import numpy as np
import PIL

import requests
import tqdm
from google import genai
from PIL import Image
load_dotenv(".env")
# Configuration
max_pixels = 1568 * 1568  # Max resolution for images

def download_collection_images():
    url = "https://data.rijksmuseum.nl/search/collection"
    headers = {"Accept": 'application/ld+json;profile="https://linked.art/ns/v1/linked-art.json"'}

    while url:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Add the current page of items to our list
        # In Linked Art search, items are usually in 'items' or 'member'
        page_items = data.get('orderedItems', [])
        
        # Check if there is a next page
        # The 'next' field is usually at the root of the JSON
        next_page_info = data.get('next')
        if next_page_info:
            url = next_page_info.get('id')
        else:
            url = None  # No more pages
        yield page_items

def extract(items):
    if not os.path.exists("rijks_images"):
        os.makedirs("rijks_images")
    for item in items:
        try:
            object_url = item['id']
            object_id= object_url.split('/')[-1]
        
            # 1. Get the individual object metadata
            # We append .json or use Accept headers for Linked Art JSON
            headers = {"Accept": "application/ld+json;profile=\"linked.art\""}
            obj_resp = requests.get(object_url, headers=headers)
            obj_data = obj_resp.json()

            # 2. Extract the Image Identifier from subject_of -> digitally_carried_by
            # This follows the Linked Art structure found in your previous response
            shows= obj_data.get("shows", [])
            [digital] = [requests.get(s.get("id")).json() for s in shows]
            if  "digitally_shown_by" not in digital:
                continue    
            [ids] = [d for d in digital["digitally_shown_by"]]
            [access] = requests.get(ids["id"]).json()["access_point"]
            img = requests.get(access["id"]).content

            artists = obj_data.get("produced_by", [])
            if artists:
                artists = artists.get("referred_to_by", [])
            if artists:
                artists = [a["content"] for a in artists]

            title = obj_data.get("identified_by", [])
            if title:
                title = [t["content"] for t in title if t["type"] == "Name"]
                #assume dutch - eng sequence
                # title = title[1::2]
            [show] = shows
            artist = artists[0]
            title = title[1]
            if img:
                filename = f"rijks_images/{object_id}.jpg"
                with open(filename, "wb") as f:
                    f.write(img)
                    print(f"Downloaded: {filename}")
            else:
                print(f"No image found for {object_id}")
            #write json file with artist and title
            with open(f"rijks_images/{object_id}.json", "w") as f:
                json.dump({"artist": artist, "title": title}, f)
            print(f"Wrote metadata for {object_id}")
        except Exception as e:
            print(f"Error processing {object_id}: {e}")

# Resize too large images
def resize_image(pil_image: PIL.Image.Image) -> None:
    """Resizes the image in-place if it exceeds max_pixels."""
    org_width, org_height = pil_image.size

    # Resize image if too large
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))




# Convert PIL image to base64 string
def pil_to_base64(pil_image: PIL.Image.Image) -> str:
    """Converts a PIL image to a base64 encoded string."""
    if pil_image.format is None:
        img_format = "PNG"
    else:
        img_format = pil_image.format

    resize_image(pil_image)

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64," + base64.b64encode(
            img_buffer.read()
        ).decode("utf-8")

    return img_data
# Convert images to a base64 string before sending it to the API
def base64_from_image(img_path: str) -> str:
    """Converts an image file to a base64 encoded string."""
    pil_image = PIL.Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"

    resize_image(pil_image)

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64," + base64.b64encode(
            img_buffer.read()
        ).decode("utf-8")

    return img_data

def compute_image_embedding(base64_img: str, _cohere_client) -> np.ndarray | None:
    """Computes an embedding for an image using Cohere's Embed-4 model."""
    try:
        api_response = _cohere_client.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            images=[base64_img],
        )

        if api_response.embeddings and api_response.embeddings.float:
            return np.asarray(api_response.embeddings.float[0])
        else:
            return None
    except Exception as e:
        print(f"Error computing embedding: {e}")
        return None

def compute_all_embeddings(image_folder: Path) -> dict[str, np.ndarray]:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.ClientV2(api_key=cohere_api_key)
    embeddings = []
    image_paths = list(image_folder.glob("*.jpg"))  
    for img in tqdm.tqdm(image_paths, "computing image embeddings"):
        base64_img = base64_from_image(str(img))
        embedding = compute_image_embedding(base64_img, co)
        embeddings.append(embedding)
    return image_paths, np.stack(embeddings)

def create_vector_store(image_paths: list[Path], embeddings: np.ndarray):
    # 1. Initialize Milvus Client
    client = MilvusClient(uri="./milvus_local.db") 

    # 2. Define the Schema
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

    # Primary Key
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, is_primary=True, max_length = 256)

    # Vector Field (Match dim to your Cohere model, e.g., 1024 for v3.0)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536)

    # 3. Create Index (HNSW is recommended for high performance)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE", # Use COSINE for Cohere v3
        index_type="IVF_FLAT",
        # params={"M": 16, "efConstruction": 500}
    )

    # 4. Create Collection
    client.create_collection(
        collection_name="cohere_collection",
        schema=schema,
        index_params=index_params
    )

    data = [{"image_path": str(image_paths[i]), "vector": embeddings[i].tolist()} for i in range(len(embeddings))]

    client.insert(collection_name="cohere_collection", data=data)



def fetch_art_data():
    # 1. Search the collection

    for items in download_collection_images():
        extract(items)



if __name__ == "__main__":
    image_paths, embeddings = compute_all_embeddings(Path("rijks_images"))
    create_vector_store(image_paths, embeddings)
    fetch_art_data()
