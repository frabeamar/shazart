import base64
from collections.abc import Iterator
import io
import json
import os
from pathlib import Path

import cohere
from dotenv import load_dotenv
import numpy as np
import PIL
import requests
import tqdm
from PIL import Image
from pymilvus import DataType, MilvusClient
from dataclasses import dataclass

load_dotenv(".env")
IMAGES = Path("images")
if not IMAGES.exists():
    IMAGES.mkdir()

MAX_PIXELS = 1568 * 1568 
@dataclass 
class ArtPiece:
    img: bytes
    metadata: dict

    @property
    def img_filename(self)->Path:
        return Path(IMAGES)/f"{self.metadata['object_id']}.jpg"

    @property
    def metadata_filename(self)->Path:
        return Path(IMAGES)/f"{self.metadata['object_id']}.json"

    def save(self):

        with open(self.img_filename, "wb") as f:
            f.write(self.img)
        with open(self.metadata_filename, "w") as f:
            json.dump(self.metadata, f)

@dataclass 
class CollectionItem:
    url: str
    
    @property
    def object_id(self):
        return self.url.split("/")[-1]


    def extract(self)->ArtPiece | None:
        # object meteadata
        headers = {"Accept": 'application/ld+json;profile="linked.art"'}
        obj_resp = requests.get(self.url, headers=headers)
        obj_data = obj_resp.json()
        # 2. Extract the Image Identifier from subject_of -> digitally_carried_by
        shows = obj_data.get("shows", [])
        [digital] = [requests.get(s.get("id")).json() for s in shows]
        if "digitally_shown_by" not in digital:
            return
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
            # assume dutch - eng sequence
            # title = title[1::2]
        artist = artists[0]
        title = title[1]
        return ArtPiece(img, {"artist": artist, "title": title, "object_id": self.object_id})






def download_collection_items() -> Iterator[CollectionItem]:
    """
    Download the image collection from the Rijksmuseum API.
    API documentation:
    https://data.rijksmuseum.nl/
    """
    url = "https://data.rijksmuseum.nl/search/collection"
    headers = {
        "Accept": 'application/ld+json;profile="https://linked.art/ns/v1/linked-art.json"'
    }

    while url:
        print(f"Fetching: {url}")
        # response is given in pages, iterate through them to get all the data
        response = requests.get(
            url, headers=headers, params={"type": "painting", "imageAvailable": True, "material": "oil paint"}
        )
        data = response.json()

        page_items: list[dict] = data.get("orderedItems", [])
        typed_page_items = [CollectionItem(p["id"])for p in page_items]

        next_page_info = data.get("next")
        if next_page_info:
            url = next_page_info.get("id")
        else:
            url = None  # No more pages
        yield from typed_page_items


def resize_image(pil_image: Image.Image) -> None:
    """Resizes the image in-place if it exceeds max_pixels."""
    org_width, org_height = pil_image.size

    if org_width * org_height > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))


def pil_to_base64(pil_image: Image.Image) -> str:
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


def compute_embeddings(image_folder: Path, max_images:int = 10) -> tuple[list[Path], np.ndarray]:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.ClientV2(api_key=cohere_api_key)
    embeddings = []
    image_paths = sorted(list(image_folder.glob("*.jpg")))[:max_images]
    breakpoint()
    for img in tqdm.tqdm(image_paths, "computing image embeddings"):
        base64_img = base64_from_image(str(img))
        embedding = compute_image_embedding(base64_img, co)
        embeddings.append(embedding)
    return image_paths, np.stack(embeddings)


def create_vector_store():
    client = MilvusClient(uri="./milvus_local.db")

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

    # add a primary key
    schema.add_field(
        field_name="image_path",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=256,
    )

    # match dim to embedding output
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536)

    # 3. index, ivf for speed, not need actually
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE", 
        index_type="IVF_FLAT",
    )

    client.create_collection(
        collection_name="cohere_collection", schema=schema, index_params=index_params
    )
    return client




def download_collection_images() -> Iterator[ArtPiece]:
    pbar = tqdm.tqdm(enumerate(download_collection_items()), desc="Downloading images")
    existing = 0
    for i, item in pbar:
        try:
            piece =   item.extract()
            if not piece:
                continue
            if not piece.img_filename.exists():
                yield piece
            else:
                existing +=1
        except:
            continue
        pbar.set_description(f"Downloaded {1+i-existing} items; {existing} already exist")


def fetch_art_data():
    
    # 1. Search the collection
    for piece in download_collection_images():
        piece.save()

"""
saved images
black and white cat: images/200105887.jpg
[PosixPath('images/200100988.jpg'), PosixPath('images/200105887.jpg'), PosixPath('images/200105889.jpg'), PosixPath('images/200105971.jpg'), PosixPath('images/200106038.jpg'), PosixPath('images/200106077.jpg'), PosixPath('images/200106078.jpg'), PosixPath('images/200106079.jpg'), PosixPath('images/200106080.jpg'), PosixPath('images/200106086.jpg')]
"""
if __name__ == "__main__":
    # fetch_art_data()
    image_paths, embeddings = compute_embeddings(IMAGES)

    client = create_vector_store()

    data = [
        {"image_path": str(image_paths[i]), "vector": embeddings[i].tolist()}
        for i in range(len(embeddings))
    ]
    client.get(ids = [d["image_path"] for d in data], collection_name ="cohere_collection")
    client.insert(collection_name="cohere_collection", data=data)
