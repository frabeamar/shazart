import os
from pathlib import Path

import cohere
import numpy as np
import streamlit as st
import tqdm

# load dotenv
from google import genai
from PIL import Image

from data import base64_from_image, compute_image_embedding

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Shazart: Vision RAG at the Rijksmuseum")
st.title("Shazart: Vision RAG at the Rijksmuseum 🖼️")

intro = """
Ever stared at a masterpiece in a museum and thought, "It’s pretty, but I can't bother to read the tedious description?" Shazart is the Shazam for the art world—with a personality. Search by title or by an image, and we’ll skip the dry, crusty textbook jargon. Instead, you get a punchy, fun retelling of the scandals, the secrets, and the "what was the artist thinking?" behind the canvas.
Ask question about the collection. The RAG augmented system will be able to answer your questions!
"""
st.text(intro)
# --- API Key Input ---
with st.sidebar:
    st.header("🔑 API Keys")

    cohere_api_key = st.text_input("Cohere API Key", type="password", key="cohere_key")
    google_api_key = st.text_input(
        "Google API Key (Gemini)", type="password", key="google_key"
    )
    if "COHERE_API_KEY" in os.environ:
        cohere_api_key = os.environ["COHERE_API_KEY"]

    if "GOOGLE_API_KEY" in os.environ:
        google_api_key = os.environ["GOOGLE_API_KEY"]

    "[Get a Cohere API key](https://dashboard.cohere.com/api-keys)"
    "[Get a Google API key](https://aistudio.google.com/app/apikey)"

    st.markdown("---")
    if not cohere_api_key:
        st.warning("Please enter your Cohere API key to proceed.")
    if not google_api_key:
        st.warning("Please enter your Google API key to proceed.")
    st.markdown("---")


# --- Initialize API Clients ---
co = None
genai_client = None
# Initialize Session State for embeddings and paths
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None

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

# --- Helper functions ---
# Some helper functions to resize images and to convert them to base64 format


# Compute embedding for an image
@st.cache_data(ttl=3600, show_spinner=False)
# Process a PDF file: extract pages as images and embed them
# Note: Caching PDF processing might be complex due to potential large file sizes and streams
# We will process it directly for now, but show progress.

# Download and embed sample images
@st.cache_data(ttl=3600, show_spinner=False)
def download_and_embed_sample_images(
    _cohere_client,
) -> tuple[list[str], np.ndarray | None]:
    """Downloads sample images and computes their embeddings using Cohere's Embed-4 model."""

    img_folder = "./paiting/"
    images = list(Path(img_folder).glob("*.jpg"))
    # Prepare folders
    os.makedirs(img_folder, exist_ok=True)

    doc_embeddings = []

    # Wrap TQDM with st.spinner for better UI integration
    with st.spinner("Downloading and embedding sample images..."):
        pbar = tqdm.tqdm(images, desc="Processing sample images")

        # Process each image
        for img_path in pbar:
            # Check if embedding already exists for this index
            try:
                # Ensure file exists before trying to embed
                if os.path.exists(img_path):
                    base64_img = base64_from_image(str(img_path))
                    emb = compute_image_embedding(
                        base64_img, _cohere_client=_cohere_client
                    )
                    if emb is not None:
                        # Placeholder to ensure list length matches paths before vstack
                        doc_embeddings.append(emb)
                    # If file doesn't exist (maybe failed download), add placeholder
            except Exception as e:
                st.error(f"Failed to embed {img_path}: {e}")
                # Add placeholder on error
    # Filter out None embeddings and corresponding paths before stacking
    filtered_paths = [
        str(path)
        for i, path in enumerate(images)
        if i < len(doc_embeddings) and doc_embeddings[i] is not None
    ]
    filtered_embeddings = [emb for emb in doc_embeddings if emb is not None]
    if filtered_embeddings:
        doc_embeddings_array = np.vstack(filtered_embeddings)
        return filtered_paths, doc_embeddings_array

    return [], None


# Search function
def search(
    question: str,
    co_client: cohere.Client,
    embeddings: np.ndarray,
    image_paths: list[str],
    max_img_size: int = 800,
) -> str | None:
    """Finds the most relevant image path for a given question."""
    if not co_client or embeddings is None or embeddings.size == 0 or not image_paths:
        st.warning(
            "Search prerequisites not met (client, embeddings, or paths missing/empty)."
        )
        return None
    if embeddings.shape[0] != len(image_paths):
        st.error(
            f"Mismatch between embeddings count ({embeddings.shape[0]}) and image paths count ({len(image_paths)}). Cannot perform search."
        )
        return None

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

        # Ensure query embedding has the correct shape for dot product
        if query_emb.shape[0] != embeddings.shape[1]:
            st.error(
                f"Query embedding dimension ({query_emb.shape[0]}) does not match document embedding dimension ({embeddings.shape[1]})."
            )
            return None

        # Compute cosine similarities
        cos_sim_scores = np.dot(query_emb, embeddings.T)

        # Get the most relevant image
        top_idx = np.argmax(cos_sim_scores)
        hit_img_path = image_paths[top_idx]
        print(f"Question: {question}")  # Keep for debugging
        print(f"Most relevant image: {hit_img_path}")  # Keep for debugging

        return hit_img_path
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


# --- Main UI Setup ---
st.subheader("📊 Load Sample Images")
if cohere_api_key and co:
    # If button clicked, load sample images into session state
    if st.button("Load Sample Images", key="load_sample_button"):
        sample_img_paths, sample_doc_embeddings = download_and_embed_sample_images(
            _cohere_client=co
        )
        if sample_img_paths and sample_doc_embeddings is not None:
            # Append sample images to session state (avoid duplicates if clicked again)
            current_paths = set(st.session_state.image_paths)
            new_paths = [p for p in sample_img_paths if p not in current_paths]

            if new_paths:
                new_indices = [
                    i for i, p in enumerate(sample_img_paths) if p in new_paths
                ]
                st.session_state.image_paths.extend(new_paths)
                new_embeddings_to_add = sample_doc_embeddings[
                    [idx for idx, p in enumerate(sample_img_paths) if p in new_paths]
                ]

                if (
                    st.session_state.doc_embeddings is None
                    or st.session_state.doc_embeddings.size == 0
                ):
                    st.session_state.doc_embeddings = new_embeddings_to_add
                else:
                    st.session_state.doc_embeddings = np.vstack(
                        (st.session_state.doc_embeddings, new_embeddings_to_add)
                    )
                st.success(f"Loaded {len(new_paths)} sample images.")

            else:
                st.info("Sample images already loaded.")

        else:
            st.error("Failed to load sample images. Check console for errors.")
else:
    st.warning("Enter API keys before proceeding.")

st.markdown("--- ")
# --- File Uploader (Main UI) ---
st.subheader("📤 Upload Your Images")
st.info(
    "Or, upload your own images. The RAG process will search across all loaded content."
)

# File uploader
uploaded_files = st.file_uploader(
    "Upload images (PNG, JPG, JPEG) or PDFs",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True,
    key="image_uploader",
    label_visibility="collapsed",
)

# Process uploaded images
if uploaded_files and co:
    st.write(f"Processing {len(uploaded_files)} uploaded images...")
    progress_bar = st.progress(0)

    # Create a temporary directory for uploaded images
    upload_folder = "uploaded_img"
    os.makedirs(upload_folder, exist_ok=True)

    newly_uploaded_paths = []
    newly_uploaded_embeddings = []

    for i, uploaded_file in enumerate(uploaded_files):
        # Check if already processed this session (simple name check)
        img_path = os.path.join(upload_folder, uploaded_file.name)
        if img_path not in st.session_state.image_paths:
            try:
                # Check file type
                file_type = uploaded_file.type
                match file_type:
                    case "application/pdf":
                        print("pdfs file not support")
                    case "image/png" | "image/jpeg":
                        # Process regular image
                        # Save the uploaded file
                        with open(img_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Get embedding
                        base_64_img = base64_from_image(img_path)
                        emb = compute_image_embedding(base_64_img, _cohere_client=co)

                        if emb is not None:
                            newly_uploaded_paths.append(img_path)
                            newly_uploaded_embeddings.append(emb)
                    case _:
                        st.warning(
                            f"Unsupported file type skipped: {uploaded_file.name} ({file_type})"
                        )

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
        # Update progress regardless of processing status for user feedback
        progress_bar.progress((i + 1) / len(uploaded_files))

    # Add newly processed files to session state
    if newly_uploaded_paths:
        st.session_state.image_paths.extend(newly_uploaded_paths)
        if newly_uploaded_embeddings:
            new_embeddings_array = np.vstack(newly_uploaded_embeddings)
            if (
                st.session_state.doc_embeddings is None
                or st.session_state.doc_embeddings.size == 0
            ):
                st.session_state.doc_embeddings = new_embeddings_array
            else:
                st.session_state.doc_embeddings = np.vstack(
                    (st.session_state.doc_embeddings, new_embeddings_array)
                )
            st.success(
                f"Successfully processed and added {len(newly_uploaded_paths)} new images."
            )
        else:
            st.warning("Failed to generate embeddings for newly uploaded images.")
    elif uploaded_files:  # If files were selected but none were new
        st.info("Selected images already seem to be processed.")


## DISPLAY IMAGES


st.subheader(" IMAGES ")


with st.container(border=True):
    if st.session_state.image_paths:
        num_images_to_show = len(st.session_state.image_paths)
        cols = st.columns(5)  # Show 5 thumbnails per row
        for i in range(num_images_to_show):
            with cols[i % 5]:
                # Add try-except for missing files during display
                try:
                    st.image(
                        st.session_state.image_paths[i],
                        width=100,
                        caption=os.path.basename(st.session_state.image_paths[i]),
                    )
                except FileNotFoundError:
                    st.error(
                        f"Missing: {os.path.basename(st.session_state.image_paths[i])}"
                    )
    else:
        st.write("No images loaded yet.")


st.markdown("---")
# st.subheader("Detecting paintings with YOLO ")

# # 1. Load your custom trained OBB model
# model = YOLO("runs/obb/train/weights/best.pt")

# with st.container(border=True):
#     if st.session_state.image_paths:
#         num_images_to_show = len(st.session_state.image_paths)
#         cols = st.columns(5)  # Show 5 thumbnails per row
#         for i in range(num_images_to_show):
#             with cols[i % 5]:
#                 # Add try-except for missing files during display
#                 try:
#                     result = model.predict(source=st.session_state.image_paths[i], conf=0.8 )
#                     x, y, xx, yy = (
#                         result[0].obb.xyxy.cpu().numpy().astype(np.int32)[0].clip(0)
#                     )
#                     breakpoint()
#                     skimage.io.imsave( "temp.jpg",result[0].orig_img[x:xx, y:yy][:, :, ::-1],      )
#                     st.image(
#                         "temp.jpg",
#                         width=result[0].orig_img.shape[0],
#                     )
#                 except FileNotFoundError:
#                     st.error("Missing file")


button = st.button(
    "Generate caption",
    key="generate caption",
    disabled=not (
        cohere_api_key
        and google_api_key
        and st.session_state.image_paths
        and st.session_state.doc_embeddings is not None
        and st.session_state.doc_embeddings.size > 0
    ),
)
if button:
    with st.spinner("Generating caption..."):
        for image in st.session_state.image_paths:
            caption_text = funny_summary(image, genai_client)
            st.markdown(f"**The better museum caption:**\n{caption_text}")

# --- Vision RAG Section (Main UI) ---


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
        and st.session_state.doc_embeddings is not None
        and st.session_state.doc_embeddings.size > 0
    ),
)

# Output Area
st.markdown("### Results")
retrieved_image_placeholder = st.empty()
answer_placeholder = st.empty()

# Run search and answer logic
if run_button:
    if (
        co
        and genai_client
        and st.session_state.doc_embeddings is not None
        and len(st.session_state.doc_embeddings) > 0
    ):
        with st.spinner("Finding relevant image..."):
            # Ensure embeddings and paths match before search
            if (
                len(st.session_state.image_paths)
                != st.session_state.doc_embeddings.shape[0]
            ):
                st.error(
                    "Error: Mismatch between number of images and embeddings. Cannot proceed."
                )
            else:
                top_image_path = search(
                    question,
                    co,
                    st.session_state.doc_embeddings,
                    st.session_state.image_paths,
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

# Footer
st.markdown("---")
st.caption(
    "Vision RAG with Cohere Embed-4 | Built with Streamlit, Cohere Embed-4, and Google Gemini 2.5 Flash"
)
