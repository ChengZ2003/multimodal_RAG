import streamlit as st

from pages.qa_demo.main_dev import qa_demo
from pages.detect_demo.main_dev import detect_demo
from pages.doc_parse_demo.main_dev import doc_parse_demo

page_names_to_funcs = {
    "研报解析": doc_parse_demo,
    "识别工具": detect_demo,
    "智能问答": qa_demo
}

selected_page = st.sidebar.selectbox(
        "选择页面", 
        page_names_to_funcs.keys(),
)
page_names_to_funcs[selected_page]()
