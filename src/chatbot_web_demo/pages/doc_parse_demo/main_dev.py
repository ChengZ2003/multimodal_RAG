import streamlit as st
import os
import logging

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ..detect_demo.table_detection.inference_pdf import detect_pdf
from ..qa_demo.summary_utils import summarize_docs, save_summary, read_summary
from ..qa_demo.data_preprocessing import (
    parse_pdf,
    convert_img_to_tables,
    convert_to_documents,
)


if "uploaded_detect_file" not in st.session_state.keys():
    st.session_state["uploaded_detect_file"] = []

if "output_file_or_folder" not in st.session_state.keys():
    st.session_state["output_file_or_folder"] = []

if "unstructured_output_dir" not in st.session_state.keys():
    st.session_state["unstructured_output_dir"] = []

DATA_DIR = "/home/gt/Chatbot_Web_Demo/doc_parse_data"
INPUT_DIR = os.path.join(DATA_DIR, "pdf-inputs")


def get_embed_model(embed_path):
    logging.info(f"Loading {embed_path}")
    embed_model = HuggingFaceEmbedding(model_name=embed_path, device="cuda:1")
    return embed_model


def load_model():
    llm = Ollama(model="qwen:14b", request_timeout=60.0)
    embed_model = get_embed_model(embed_path="/home/gt/Chatbot_Web_Demo/model/bge-m3")
    Settings.llm = llm
    Settings.embed_model = embed_model


def clear_dirs():
    # make sure the directories exist and no files are in them
    # So this is a bit of a hack, but it works for now
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    else:
        for file in os.listdir(INPUT_DIR):
            os.remove(os.path.join(INPUT_DIR, file))


def process_data():
    file = st.session_state["uploaded_detect_file"]
    if file:
        clear_dirs()
        file_name = file.name
        filepath = os.path.join(INPUT_DIR, file_name)
        with open(filepath, "wb") as f:
            f.write(file.getbuffer())
        os.makedirs(os.path.join(DATA_DIR, file_name.split(".")[0]), exist_ok=True)
        output_dir = os.path.join(DATA_DIR, file_name.split(".")[0])
        unstructured_output_dir = os.path.join(output_dir, "unstructured")

        detect_pdf(filepath, output_dir)

        raw_docs = parse_pdf(
            filepath,
            extract_image_block_output_dir=os.path.join(
                unstructured_output_dir, "images"
            ),
            extract_images_in_pdf=True,
        )
        docs = convert_img_to_tables(raw_docs, unstructured_output_dir)
        documents, text_seq = convert_to_documents(docs)
        summary = summarize_docs(text_seq)
        save_summary(summary, unstructured_output_dir)

        st.session_state["output_file_or_folder"] = output_dir
        st.session_state["unstructured_output_dir"] = unstructured_output_dir
        st.success("解析成功！")


def visulize_result():
    if "output_file_or_folder" in st.session_state.keys():
        output_file_or_folder = st.session_state["output_file_or_folder"]
    if "unstructured_output_dir" in st.session_state.keys():
        unstructured_output_dir = st.session_state["unstructured_output_dir"]
    #summary = read_summary(unstructured_output_dir)

    if output_file_or_folder and unstructured_output_dir:
        summary = read_summary(unstructured_output_dir)
        #st.write(f"{output_file_or_folder}")
        st.write(summary)
        all_files = os.listdir(output_file_or_folder)
        all_images = [file for file in all_files if file.endswith(".png")]
        all_files_path = [
            os.path.join(output_file_or_folder, file)
            for file in all_images
        ]
        st.image(all_files_path, width=200, caption=all_images)


def upload_data():
    uploaded_detect_file = st.file_uploader(
        "请上传您的文档",
        type=["pdf"],
    )

    st.session_state["uploaded_detect_file"] = uploaded_detect_file
    if uploaded_detect_file:
        with st.spinner("正在解析..."):
            process_data()


def doc_parse_demo():
    st.header("研报解析")
    with st.sidebar:
        st.image("/home/gt/Chatbot_Web_Demo/assets/logo.jpg", use_column_width=True)

    upload_data()
    load_model()
    visulize_result()
