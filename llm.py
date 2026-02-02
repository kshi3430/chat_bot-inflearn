import streamlit as st
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
from langsmith import Client
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import answer_examples
load_dotenv()


store = {}


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()       
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



def get_retriever():
    embeddings = UpstageEmbeddings(
        model="solar-embedding-1-large"
    )

    database = PineconeVectorStore(
        index_name="tax-upstage-index",
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],)
    retriever = database.as_retriever(search_kwargs={"k": 3})
    return retriever

    



def get_llm():
    model_name = "gpt-3.5-upstage"  # 실제 사용 가능한 모델 이름
    api_key = os.getenv("UPSTAGE_API_KEY")

    # ChatUpstage 생성 시 필수값만 넣음
    llm = ChatUpstage(
        model=model_name,
        api_key=api_key
    )
    return llm


def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain
    



def get_ai_response(user_message):
    rag_chain = get_rag_chain()

    ai_response = rag_chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": "abc123"}},
    )

    return ai_response