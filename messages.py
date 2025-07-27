from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id="moonshotai/Kimi-K2-Instruct",
        huggingfacehub_api_token="your_api",
    )
)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me abouut LangChain in 1 sentence."),  
]
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)