import urllib
import uuid
from base64 import b64decode

import typer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryStore
from langchain_groq.chat_models import (
    ChatGroq,  # they use an LPU to run their models -> faster
)
from lanchain.retrievers.multivector_retriever import MultiVectorRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from unstructured.partition.pdf import partition_pdf

from model import load_gemini_chat, load_gemini_embeddings


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )



app = typer.Typer(pretty_exceptions_show_locals=False)

urllib.request.urlretrieve("https://arxiv.org/pdf/1706.03762", "attention.pdf")


def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def main():
    chunks = partition_pdf(
        filename="attention.pdf",
        infer_table_structure=True,  # extract tables
        strategy="hi_res",  # mandatory to infer tables
        extract_image_block_types=[
            "Image"
        ],  # Add 'Table' to list to extract image of tables
        # image_output_dir_path=output_path,   # if None, images and tables will saved in base64
        extract_image_block_to_payload=True,  # if true, will extract base64 for API usage
        chunking_strategy="by_title",  # or 'basic'
        max_characters=10000,  # defaults to 500
        combine_text_under_n_chars=2000,  # defaults to 0
        new_after_n_chars=6000,
    )
    # separate tables from texts
    tables = []
    texts = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)

    images = get_images_base64(chunks)

    # Prompt
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")

    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # Summarize text
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    # Summarize tables
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

    prompt_template = """Describe the image in detail. For context,
                    the image is part of a research paper explaining the transformers
                    architecture. Be specific about graphs, such as bar plots."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | load_gemini_chat() | StrOutputParser()

    image_summaries = chain.batch(images)

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=load_gemini_embeddings(),
        persist_directory="./chroma_multimodal_db",
    )

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]})
        for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]})
        for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | load_gemini_chat()
        | StrOutputParser()
    )

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | load_gemini_chat()
            | StrOutputParser()
        )
    )

main()
