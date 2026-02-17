import os
from pathlib import Path

import cohere
import numpy as np
import skimage
import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image
from pymilvus import MilvusClient
from ultralytics import YOLO

from data import COLLECTION_NAME, base64_from_image, compute_image_embedding

co = None
genai_client = None
cohere_api_key = (
    os.environ["COHERE_API_KEY"] if "COHERE_API_KEY" in os.environ else None
)
google_api_key = (
    os.environ["GOOGLE_API_KEY"] if "COHERE_API_KEY" in os.environ else None
)

load_dotenv(".env")
TEST_FOLDER = Path("test_images")


def intro():
    st.set_page_config(
        layout="wide", page_title="Shazart: Vision RAG at the Rijksmuseum"
    )
    st.title("Shazart: Vision RAG at the Rijksmuseum 🖼️")

    intro = """
    Ever stared at a masterpiece in a museum and thought, "It’s pretty, but I can't bother to read the tedious description?" Shazart is the Shazam for the art world—with a personality. Search by title or by an image, and we’ll skip the dry, crusty textbook jargon. Instead, you get a punchy, fun retelling of the scandals, the secrets, and the "what was the artist thinking?" behind the canvas.
    Ask question about the collection. The RAG augmented system will be able to answer your questions!
    """
    st.text(intro)


def api_sidebar():
    global cohere_api_key, google_api_key, co, genai_client
    with st.sidebar:
        st.header("🔑 API Keys")
        if not cohere_api_key:
            cohere_api_key = st.text_input(
                "Cohere API Key", type="password", key="cohere_key"
            )
        if not google_api_key:
            google_api_key = st.text_input(
                "Google API Key (Gemini)", type="password", key="google_key"
            )
        "[Get a Cohere API key](https://dashboard.cohere.com/api-keys)"
        "[Get a Google API key](https://aistudio.google.com/app/apikey)"

        st.markdown("---")
        if not cohere_api_key:
            st.warning("Please enter your Cohere API key to proceed.")
        if not google_api_key:
            st.warning("Please enter your Google API key to proceed.")
        st.markdown("---")

    if cohere_api_key and google_api_key:
        try:
            co = cohere.ClientV2(api_key=cohere_api_key)
            st.sidebar.success("Cohere Client Initialized!")
        except Exception as e:
            st.sidebar.error(f"Cohere Initialization Failed: {e}")

        try:
            genai_client = genai.Client(api_key=google_api_key)
            st.sidebar.success("Gemini Client Initialized!")
        except Exception as e:
            st.sidebar.error(f"Gemini Initialization Failed: {e}")
    else:
        st.info("Enter your API keys in the sidebar to start.")


# Initialize Session State for embeddings and paths
if "image_paths" not in st.session_state:
    st.session_state.image_paths = set()
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None


def model_info():
    # Information about the models
    with st.expander("ℹ️ About the models used"):
        st.markdown("""
        ### Cohere Embed-4
        
        Cohere's Embed-4 is a state-of-the-art multimodal embedding model designed for enterprise search and retrieval. 
        It enables:
        
        - **Multimodal search**: Search text and images together seamlessly
        - **High accuracy**: State-of-the-art performance for retrieval tasks
        - **Efficient embedding**: Process complex images like charts, graphs, and infographics
        
        The model processes images without requiring complex OCR pre-processing and maintains the connection between visual elements and text.
        
        ### Google Gemini 2.5 Flash
        
        Gemini 2.5 Flash is Google's efficient multimodal model that can process text and image inputs to generate high-quality responses.
        It's designed for fast inference while maintaining high accuracy, making it ideal for real-time applications like this RAG system.
        
        ### YOLO 8n
            Is trained from scratch. It's training data are painting from the museum after applying an homography and added to colored background
                    
                    """)


def add_image_to_db(
    image_paths: list[Path], client: MilvusClient, _cohere_client: cohere.ClientV2
):
    filtered = list(
        filter(
            lambda x: client.get(ids=[x], collection_name=COLLECTION_NAME) is None,
            image_paths,
        )
    )

    embeddings = []
    for img_path in filtered:
        base64_img = base64_from_image(str(img_path))
        emb = compute_image_embedding(base64_img, _cohere_client=_cohere_client)
        embeddings.append(emb)

    data = [
        {"image_path": str(im), "vector": emb}
        for im, emb in zip(image_paths, embeddings)
    ]
    client.insert(collection_name=COLLECTION_NAME, data=data)


# Download and embed sample images
@st.cache_data(ttl=3600, show_spinner=False)
def embed_images(
    images: list[Path],
    _cohere_client,
):
    """Downloads sample images and computes their embeddings using Cohere's Embed-4 model."""
    global vector_db
    with st.spinner("Downloading and embedding images..."):
        add_image_to_db(images, vector_db, _cohere_client)
    st.session_state.image_paths.update(set(images))


# Search function
def search(
    question: str,
    co_client: cohere.Client,
) -> str | None:
    """Finds the most relevant image path for a given question."""
    global vector_db

    try:
        # Compute the embedding for the query
        api_response = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[question],
        )

        if not api_response.embeddings or not api_response.embeddings.float:
            st.error("Failed to get query embedding.")
            return None

        query_emb = np.asarray(api_response.embeddings.float[0])

        [results] = vector_db.search(
            collection_name=COLLECTION_NAME,
            data=[query_emb],
            anns_field="vector",
            limit=1,
            output_fields=["image_path"],
        )
        return results[0]["id"]
    except Exception as e:
        st.error(f"Error during search: {e}")
        return None


def validate(img_path: str, gemini_client) -> list:
    missing = []
    if not gemini_client or not img_path or not os.path.exists(img_path):
        if not gemini_client:
            missing.append("Gemini client")
        if not img_path:
            missing.append("Image path")
        elif not os.path.exists(img_path):
            missing.append(f"Image file at {img_path}")
    return missing


def call_with_prompt(prompt: str, img_path: str, gemini_client) -> str:
    try:
        img = Image.open(img_path)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=[prompt, img]
        )

        llm_answer = response.text
        print("LLM Answer:", llm_answer)  # Keep for debugging
        return llm_answer
    except Exception as e:
        st.error(f"Error during answer generation: {e}")
        return f"Failed to generate answer: {e}"


# Answer function
def answer(question: str, img_path: str, gemini_client) -> str:
    """Answers the question based on the provided image using Gemini."""
    missing = validate(img_path, gemini_client)
    if missing:
        return f"Answering prerequisites not met ({', '.join(missing)} missing or invalid)."

    prompt = f"""Answer the question based on the following image. Be as elaborate as possible giving extra relevant information.
Don't use markdown formatting in the response.
Please provide enough context for your answer.

Question: {question}"""
    return call_with_prompt(prompt, img_path, gemini_client)


def funny_summary(img_path: str, gemini_client):
    missing = validate(img_path, gemini_client)
    if missing:
        return f"Answering prerequisites not met ({', '.join(missing)} missing or invalid)."

    prompt = """Provide a funny and engaging summary of the content in the following image.
    Provide a short and witty description of the life events of the author that are relevant to understand the image. 
    Format in the following structure:
    <Painting title> - <Artist>
    [Painting description]
    [Artist llife events which are relevant to the painting ]
    """
    return call_with_prompt(prompt, img_path, gemini_client)


def update_globals(img_paths: list[Path], embeddings: np.ndarray):
    assert len(img_paths) == len(embeddings)
    st.session_state.image_paths.add(img_paths)

    st.session_state.doc_embeddings = np.vstack(
        (st.session_state.doc_embeddings, embeddings)
    )


def load_sample_images():
    global vector_db, cohere_api_key, co
    sample_images = list(Path(TEST_FOLDER).glob("*.jpg"))
    # --- Main UI Setup ---
    st.subheader("📊 Load Sample Images")
    load_images = st.button("Load Sample Images", key="load_sample_button")
    if cohere_api_key and co:
        # If button clicked, load sample images into session state
        new_paths = [s for s in sample_images if s not in st.session_state.image_paths]
        if load_images:
            if new_paths:
                embed_images(new_paths, _cohere_client=co)
                st.success(f"Loaded {len(new_paths)} sample images.")
            else:
                st.info("Sample images already loaded.")

    else:
        st.warning("Enter API keys before proceeding.")


def upload_images(vector_db: MilvusClient):
    st.markdown("--- ")
    st.subheader("📤 Upload Your Images")
    st.info(
        "Or, upload your own images. The RAG process will search across all loaded content."
    )

    uploaded_files = st.file_uploader(
        "Upload images (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="image_uploader",
        label_visibility="collapsed",
    )

    # Process uploaded images
    if uploaded_files and co:
        st.write(f"Processing {len(uploaded_files)} uploaded images...")
        # progress_bar = st.progress(0)

        upload_folder = Path("uploaded_img")
        os.makedirs(upload_folder, exist_ok=True)

        new_paths = [s for s in uploaded_files if s not in st.session_state.image_paths]
        valid_paths = []
        for i, uploaded_file in enumerate(new_paths):
            try:
                if uploaded_file.type not in {"image/png", "image/jpeg"}:
                    st.warning(
                        "File format not supported. Skipping {}".format(
                            uploaded_file.name
                        )
                    )
                    continue
                else:
                    img_path = upload_folder / uploaded_file.name
                    with open(img_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    valid_paths.append(img_path)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            # progress_bar.progress((i + 1) / len(uploaded_files))
        embed_images(valid_paths, co)

        st.success(f"Successfully processed and added {len(new_paths)} new images.")
    elif uploaded_files and not co:
        st.warning("Failed to generate embeddings for newly uploaded images.")


def display_images():
    st.subheader("Loaded images")
    images = list(st.session_state.image_paths)
    with st.container(border=True):
        if st.session_state.image_paths:
            num_images_to_show = len(st.session_state.image_paths)
            cols = st.columns(5)  # Show 5 thumbnails per row
            for i in range(num_images_to_show):
                with cols[i % 5]:
                    # Add try-except for missing files during display
                    try:
                        st.image(images[i], width=100, caption=images[i].name)
                    except FileNotFoundError:
                        st.error(f"Missing: {images[i].name}")
        else:
            st.write("No images loaded yet.")


def display_paintings():
    model = YOLO("yolo/model.pt")

    st.subheader("Detected paintings")
    images = list(st.session_state.image_paths)
    with st.container(border=True):
        if st.session_state.image_paths:
            num_images_to_show = len(st.session_state.image_paths)
            cols = st.columns(5)  # Show 5 thumbnails per row
            for i in range(num_images_to_show):
                with cols[i % 5]:
                    # Add try-except for missing files during display
                    try:
                        result = model.predict(source=images[i], conf=0.7)
                        if result and len(result[0].obb.xyxy) > 0:
                            x, y, xx, yy = (
                                result[0]
                                .obb.xyxy.cpu()
                                .numpy()
                                .astype(np.int32)[0]
                                .clip(0)
                            )
                            skimage.io.imsave(
                                "temp.jpg",
                                result[0].orig_img[y:yy, x:xx][:, :, ::-1],
                            )
                            st.image(
                                "temp.jpg",
                                width=result[0].orig_img.shape[0],
                            )
                    except FileNotFoundError:
                        st.error(f"Missing: {images[i].name}")
        else:
            st.write("No images loaded yet.")


st.markdown("---")


def generate_caption():
    generate_caption_button = st.button(
        "Generate caption",
        key="generate caption",
        disabled=not (
            cohere_api_key and google_api_key and st.session_state.image_paths
        ),
    )
    if generate_caption_button:
        with st.spinner("Generating caption..."):
            for image in st.session_state.image_paths:
                caption_text = funny_summary(image, genai_client)
                st.markdown(f"**The better museum caption:**\n{caption_text}")

    st.subheader("❓ Ask a Question")

    if not st.session_state.image_paths:
        st.warning("Please load sample images or upload your own images first.")
    else:
        st.info(
            f"Ready to answer questions about {len(st.session_state.image_paths)} images. The rijkmuseum collection is also included!"
        )

    question = st.text_input(
        "Ask a question about the loaded images:",
        key="main_question_input",
        placeholder="which painting depicts van gogh?",
        disabled=not st.session_state.image_paths,
    )

    run_button = st.button(
        "Run Vision RAG",
        key="main_run_button",
        disabled=not (
            cohere_api_key
            and google_api_key
            and question
            and st.session_state.image_paths
        ),
    )

    # Output Area
    st.markdown("### Results")
    retrieved_image_placeholder = st.empty()
    answer_placeholder = st.empty()

    # Run search and answer logic
    if run_button:
        if co and genai_client:
            with st.spinner("Finding relevant image..."):
                # Ensure embeddings and paths match before search
                top_image_path = search(
                    question,
                    co,
                )

                if top_image_path:
                    caption = f"Retrieved content for: '{question}' (Source: {os.path.basename(top_image_path)})"

                    retrieved_image_placeholder.image(
                        top_image_path, caption=caption, use_container_width=True
                    )

                    with st.spinner("Generating answer..."):
                        final_answer = answer(question, top_image_path, genai_client)
                        answer_placeholder.markdown(f"**Answer:**\n{final_answer}")
                else:
                    retrieved_image_placeholder.warning(
                        "Could not find a relevant image for your question."
                    )
                    answer_placeholder.text("")  # Clear answer placeholder
        else:
            # This case should ideally be prevented by the disabled state of the button
            st.error(
                "Cannot run RAG. Check API clients and ensure images are loaded with embeddings."
            )


def footer():
    # Footer
    st.markdown("---")
    st.caption(
        "Vision RAG with Cohere Embed-4 | Built with Streamlit, Cohere Embed-4, and Google Gemini 2.5 Flash"
    )


vector_db = MilvusClient(uri="./milvus_local.db")


intro()
api_sidebar()
model_info()
load_sample_images()
upload_images(vector_db)
display_images()
display_paintings()
generate_caption()
footer()
