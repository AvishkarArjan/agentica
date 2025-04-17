from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config.settings import HF_API_KEY, hf_model

class Chain(object):
    """docstring for Chain."""
    def __init__(self):
        super(Chain, self).__init__()

        # HuggingFace model
        self.llm = HuggingFaceHub(
            repo_id=hf_model, 
            model_kwargs={"temperature": 0.3}, 
            huggingfacehub_api_token=HF_API_KEY)

        # Summarization prompt
        self.summarize_prompt = PromptTemplate.from_template("""
        Summarize the following content in simple, concise points:\n\n{context}
        """)

        # Synthesis prompt
        self.synthesis_prompt = PromptTemplate.from_template("""
        Based on the following summarized documents, answer the question below:\n
        Summaries:\n{summaries}\n\n
        Question: {question}
        """)

        self.query_prompt = PromptTemplate.from_template("""
        You are a helpful assistant who rewrites research questions into 1-2 effective Google search queries.

        Original question:
        {question}

        Simply return the Search queries:
        """)

    def generate_search_queries(self, question: str) -> str:
        query_chain = self.query_prompt | self.llm
        return query_chain.invoke({"question": question})

    def get_summarize_chain(self):
        summarize_chain = LLMChain(llm=self.llm, prompt=self.summarize_prompt)
        return summarize_chain

    def get_synthesis_chain(self):
        synthesis_chain = LLMChain(llm=self.llm, prompt=self.synthesis_prompt)
        return synthesis_chain