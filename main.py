from agent.tools import WebAgent
from agent.chains import Chain
import gradio as gr

from pprint import pprint

webagent = WebAgent()
chain = Chain()

tavily_tool = webagent.get_search_tool()

summarize_chain = chain.get_summarize_chain()
synthesis_chain = chain.get_synthesis_chain()

# Define the function for the chat interface
def research_chat(message, history):
    # Step 1: Web search
    search_content = tavily_tool.run(message)

    # Step 2: Summarization
    summary = summarize_chain.run({"context": search_content})

    # Step 3: Synthesis
    answer = synthesis_chain.run({"summaries": summary, "question": message})

    return answer

# Create the Gradio ChatInterface
chat = gr.ChatInterface(
    fn=research_chat,
    title="🔍 LangChain Research Assistant",
    description="Ask research-level questions and get deep answers from a LangChain-powered AI.",
    examples=[
        "What are the latest breakthroughs in CRISPR gene editing?",
        "Explain the impact of renewable energy adoption in Europe.",
        "What are the risks and benefits of using LLMs in education?"
    ],
    theme="soft",
)

if __name__ == "__main__":
    chat.launch()
