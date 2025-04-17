from langchain.tools import Tool
from tavily import TavilyClient
from config.settings import TAVILY_API_KEY

class WebAgent(object):
    """docstring for WebAgent."""
    def __init__(self, max_results=1):
        super(WebAgent, self).__init__()

        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        self.max_results = max_results

    def search(self, query: str) -> str:
        return self.tavily.search(
            query=query, 
            search_depth="basic", 
            max_results=self.max_results
            )

    def web_search_tool_fn(self, query: str) -> str:
        results = self.search(query=query)
        contents = []
        for r in results.get("results", []):
            contents.append(r["content"])
        return "\n\n".join(contents)

    def get_search_tool(self):
        tavily_tool = Tool(
            name="TavilySearch",
            func=self.web_search_tool_fn,
            description="Useful for researching topics by searching the web and retrieving relevant content."
        )

        return tavily_tool
