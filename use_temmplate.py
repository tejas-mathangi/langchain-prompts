from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    huggingfacehub_api_token="your_huggingface_api_token_here",
)
model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_input = st.selectbox("Select a Research paper:", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners"])
style_input = st.selectbox("Select an Explanation   style:", ["Formal", "Informal", "Technical"])
length_input = st.slider("Select response length:", 50, 500, 100)

#template
template = PromptTemplate(
    input_variables=["paper", "style", "length"],
    template="""Please provide a summary of the research paper titled '{paper}' in a {style} style with a response length of approximately {length} words.
    1.Mathematical Details:
        -Include key equations and their significance.
        -Explain the mathematical concepts in a way that is accessible to a reader with a background in {style} style.
    2.Analogies:
        -Use analogies to clarify complex concepts.
        -Ensure the analogies are relevant to the {style} style.
    3.Implementation Details:
        -Discuss how the methods can be implemented in practice.
        -Provide code snippets or pseudocode if applicable.
    4.Applications:
        -Highlight potential applications of the research.
        -Discuss how these applications relate to the {style} style.
    """ 
)
#place holders for the template
prompt = template.invoke({
    "paper": paper_input,
    "style": style_input,
    "length": length_input
    })

if st.button("Generate Response"):
    result = model.invoke(prompt)
    st.write(result.content)