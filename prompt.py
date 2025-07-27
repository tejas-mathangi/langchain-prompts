from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
import streamlit as st

# HuggingFaceH4/zephyr-7b-beta
llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    huggingfacehub_api_token="api_key",
)

model = ChatHuggingFace(llm=llm)

st.header("HuggingFace Chat Model Reseach Tool")

user_input = st.text_input("Enter your question here:")

if st.button("Submit"):
    response = model.invoke(user_input)
    st.write(response.content)



