import streamlit as st
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
from langsmith import Client
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from llm import get_ai_response 

load_dotenv()
 
st.set_page_config(
    page_title="김성현 챗봇",
    page_icon="💩",
)

st.title("💩소득세 챗봇")
st.caption("소득세에 관한 무엇이든 물어보세요..!!!")


if "messages_list" not in st.session_state:
    st.session_state.messages_list = []

print(f"before == {st.session_state.messages_list}")
for message in st.session_state.messages_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input(placeholder="소득세에 관한 궁금한것을 물어보셈"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages_list.append(
        {"role": "user", "content": user_question}
    )

    with st.spinner("AI가 답변을 작성하는 중..."):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
        st.session_state.messages_list.append(
            {"role": "ai", "content": ai_message}
        )

print(f"after == {st.session_state.messages_list}")
