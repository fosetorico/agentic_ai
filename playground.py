import openai
from agno.agent import Agent
import agno.api
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground, serve_playground_app
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
agno.api = os.getenv("PHI_API_KEY")  # AGNO_API_KEY

# Web Search Agent ============
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    markdown=True,
)

# Financial Agent =============
finance_agent = Agent(
    name = "Finance AI Agent",
    model = Groq(id="llama3-70b-8192"),
    tools = [
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        ),
    ],
    instructions = ["Use tables to display the data"],
    markdown = True,
)

# Serve the Playground ============
app = Playground(agents=[finance_agent, web_search_agent]).get_app()


if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
