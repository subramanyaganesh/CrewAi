from crewai import Agent, Task, Crew, Process
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain.tools import tool
from langchain_chroma import Chroma
from typing import List
import os
from crewai import LLM
llm_azure_agent = LLM(
    model="gpt-4o-mini",
    api_key="",
    base_url="",
    api_version="2023-05-15",
    azure=True
)
embedding_function = AzureOpenAIEmbeddings(
azure_deployment="text-embedding-ada-002",  # e.g. ""
    openai_api_key="",
    azure_endpoint="",
    openai_api_version="2023-05-15"
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  # e.g. "gpt-4"
    api_key="",
    azure_endpoint="",
    openai_api_version="2023-05-15"
)
current_directory = os.getcwd()
persist_directory = os.path.join(current_directory, "chroma_db")


print(persist_directory)


class NewsDatabase:
    def __init__(self, file_paths: List[str], chunk_size: int = 1000,dont_load:bool =True, chunk_overlap: int = 200):
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dont_load=dont_load
        
    def search(self, query: str):
        try:
            # Initialize Chroma with embedding function
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function
            )
            # Perform similarity search
            results = vectorstore.similarity_search(query)
            
            return results
        except Exception as e:
            return f"Error searching Knowledge Base database: {str(e)}"
    
    def index_content(self):
        from Genie import Genie
                
        Genie(file_path=self.file_paths[0],
              embedding_function=embedding_function,
              llm=llm,
              dont_load=self.dont_load,
              persist_directory=persist_directory)

# Create a tool wrapper for the search method
def create_search_tool(news_db: NewsDatabase):
    @tool("Knowledge Base DB Tool")
    def search_tool(query: str) -> str:
        """
        Search Knowledge Base from JSONL files and process their texts.
        Args:
            query (str): The search query string
        Returns:
            str: Search results as a formatted string
        """
        results = news_db.search(query)
        return results
    
    return search_tool

# Create the agents with the properly wrapped tool
def setup_agents(news_db: NewsDatabase):
    search_tool = create_search_tool(news_db)
    
    data_search_agent = Agent(
        role='Knowledge Base Researcher',
        goal='Search and analyze Knowledge Base to extract key information regarding a given question along with their metadata information of urls',
        backstory="""Expert Knowledge Base analyst with deep experience in extracting valuable insights 
        from Knowledge Base. Skilled at identifying important details.""",
        tools=[search_tool],
        allow_delegation=True,
        verbose=True,
        llm=llm_azure_agent
    )

    # writer_agent = Agent(
    #     role='Content Synthesizer',
    #     goal='Create comprehensive, well-structured summaries from analyzed Knowledge Base content include the metadata information of URL',
    #     backstory="""Experienced content creator specializing in synthesizing complex 
    #     information into clear, engaging narratives. Expert at identifying connections 
    #     between different pieces of information.""",
    #     tools=[search_tool],
    #     allow_delegation=True,
    #     verbose=True,
    #     llm=llm_azure_agent
    # )
    
    return data_search_agent

def create_answer_crew(query: str, news_db: NewsDatabase) -> Crew:
    data_search_agent = setup_agents(news_db)
    
    search_task = Task(
        description=f"""
        1. Search for knowledge base related to: {query}
        2. Analyze the document to find the relevant information
        3. Create a structured list of key points for each article
        """,
        agent=data_search_agent,
        expected_output="Detailed summary of search results"
    )

    # synthesis_task = Task(
    #     description=f"""
    #     1. Review all key points identified in the Knowledge Base search
    #     2. For each major topic:
    #        - Verify the information using the search tool
    #        - Analyze the context and significance
    #        - Identify any patterns or trends
    #     3. Create a detailed summary that:
    #        - Presents information in a logical order
    #        - Highlights key developments
    #        - Explains implications
    #        - Connects related information
    #     """,
    #     agent=writer_agent,
    #     context=[search_task],
    #     expected_output="Consolidate all the findings and return the answer to be displayed to the user"
    # )

    return Crew(
        agents=[data_search_agent],
        tasks=[search_task],
        process=Process.sequential,
        manager_llm=llm_azure_agent
    )


def process_query(query: str) -> str:
     # Initialize and index content
    print("|||||||||||||||||||||||||||||||||||||||||||||||")
    print('this is the entryyyyy')
    print("|||||||||||||||||||||||||||||||||||||||||||||||")
    stanford_db = NewsDatabase(file_paths=[
        './fingate.jsonl',
    ])
    stanford_db.index_content()
    
    # Create and run crew
    crew = create_answer_crew(query,stanford_db)
    result = crew.kickoff()
    return str(result)

# Example usage
if __name__ == "__main__":
    # Initialize the database
    news_db = NewsDatabase(file_paths=[
        './fingate.jsonl'
    ])
    news_db.index_content()
    
    # Create and run the crew
    query = "Give me 2024 holidays"
    crew = create_answer_crew(query, news_db)
    result = crew.kickoff()
    print(result)