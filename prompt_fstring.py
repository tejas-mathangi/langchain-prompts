from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    huggingfacehub_api_token="your_api_token_here",
)
model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_input = st.selectbox("Select a Research paper:", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners"])
style_input = st.selectbox("Select an Explanation   style:", ["Formal", "Informal", "Technical"])
length_input = st.slider("Select response length:", 50, 500, 100)

if st.button("Generate Response"):
    user_input = f"Paper: {paper_input}, Style: {style_input}, Length: {length_input}"
    response = model.invoke(user_input)
    st.write(response.content)