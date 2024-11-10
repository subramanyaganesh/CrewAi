from typing import List, Dict, Any, Generator, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
import json
import time
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import os
import gc
import shutil


class Genie:
    """
    An optimized class for loading, processing, and querying large JSONL files using LangChain components.
    """
    
    def __init__(
        self,
        dont_load: bool,
        file_path: str,
        text_fields: List[str] = None,
        metadata_fields: List[str] = None,
        batch_size: int = 100,
        rate_limit: float = 0.1,  # Time in seconds between batches
        max_workers: int = 4,
        persist_directory: Optional[str] = None,
        llm: AzureChatOpenAI = None,
        embedding_function: AzureOpenAIEmbeddings = None
        
    ):
        if not llm or not isinstance(llm, AzureChatOpenAI):
            raise ValueError("Name is required and must be a string.")
        if not embedding_function or not isinstance(embedding_function, AzureOpenAIEmbeddings):
            raise ValueError("Name is required and must be a string.")
        """
        Initialize the Genie with optimized JSONL processing capabilities.
        
        Args:
            file_path (str): Path to the JSONL file
            text_fields (List[str]): List of fields to combine as the main text content
            metadata_fields (List[str]): List of fields to include as metadata
            batch_size (int): Number of documents to process in each batch
            rate_limit (float): Time in seconds to wait between batches
            max_workers (int): Maximum number of threads for parallel processing
            persist_directory (str): Directory to persist ChromaDB data
        """
        # Setup logging
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        
        self.file_path = file_path
        self.text_fields = text_fields or ["text"]
        self.metadata_fields = metadata_fields or ["url"]
        self.batch_size = batch_size
        self.rate_limit = rate_limit
        self.max_workers = max_workers
        self.persist_directory = persist_directory
        self.llm = llm
        self.embedding_function = embedding_function
        self.dont_load=dont_load
        
        # Create persist directory if it doesn't exist
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB with persistence if specified
        self.vectordb = self._initialize_vectordb()
        
        # Load and process the initial data from all files
        if not self.dont_load:
            # if os.path.exists("chroma_db"):
            #     shutil.rmtree("chroma_db")
            # else:
            #     print("Folder does not exist.")
            self._process_initial_data()
        
        # Initialize the QA chain
        self.genie = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever()
        )

    def _initialize_vectordb(self):
        """Initialize ChromaDB with or without persistence."""
        if self.persist_directory:
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
        return None

    def _process_initial_data(self):
        """Process the initial JSONL file in batches."""
        self.logger.info("Starting initial data processing...")
        
        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(self.file_path, 'r', encoding='utf-8'))
        
        documents = self._load_documents_in_batches(total_lines)
        if not self.vectordb:
            # Initialize vectordb with first batch if not persistent
            self.vectordb = Chroma.from_documents(
                next(documents),
                self.embedding_function,
                persist_directory=self.persist_directory
            )
        
        # Process remaining batches
        for doc_batch in documents:
            if doc_batch:  # Skip empty batches
                self.vectordb.add_documents(doc_batch)
                time.sleep(self.rate_limit)  # Rate limiting between batches
                gc.collect()  # Help manage memory

    def _parse_jsonl_line(self, line: str) -> Optional[Document]:
        """Parse a single JSONL line into a Document object."""
        try:
            entry = json.loads(line.strip())
            
            text_content = " ".join(
                str(entry.get(field, "")) for field in self.text_fields
            ).strip()
            
            metadata = {
                field: entry.get(field)
                for field in self.metadata_fields
                if field in entry
            }
            
            return Document(page_content=text_content, metadata=metadata) if text_content else None
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error parsing JSONL line: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Error processing line: {e}")
            return None

    def _load_documents_in_batches(self, total_lines: int) -> Generator[List[Document], None, None]:
        """
        Load and process JSONL file in batches using parallel processing.
        
        Args:
            total_lines (int): Total number of lines in the file
            
        Yields:
            List[Document]: Batch of processed documents
        """
        batch = []
        with tqdm(total=total_lines, desc="Processing documents") as pbar:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit batch of lines for parallel processing
                    future_to_line = {
                        executor.submit(self._parse_jsonl_line, line): line 
                        for line in itertools.islice(file, self.batch_size)
                    }
                    
                    while future_to_line:
                        # Process completed futures
                        for future in as_completed(future_to_line):
                            line = future_to_line.pop(future)
                            try:
                                doc = future.result()
                                if doc:
                                    batch.append(doc)
                            except Exception as e:
                                self.logger.error(f"Error processing line: {e}")
                            pbar.update(1)
                            
                            # Submit new line for processing if available
                            next_line = next(file, None)
                            if next_line is not None:
                                new_future = executor.submit(self._parse_jsonl_line, next_line)
                                future_to_line[new_future] = next_line
                        
                        # Yield batch if it reaches batch_size
                        if len(batch) >= self.batch_size:
                            split_texts = self.text_split(batch)
                            yield split_texts
                            batch = []
                            gc.collect()  # Help manage memory
                    
                    # Yield remaining documents
                    if batch:
                        split_texts = self.text_split(batch)
                        yield split_texts

    @staticmethod
    def text_split(documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)

    def add_documents(self, new_jsonl_path: str) -> None:
        """
        Add additional documents from another JSONL file with batch processing.
        
        Args:
            new_jsonl_path (str): Path to the new JSONL file
        """
        self.logger.info(f"Adding documents from {new_jsonl_path}")
        original_path = self.file_path
        self.file_path = new_jsonl_path
        
        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(new_jsonl_path, 'r', encoding='utf-8'))
        
        # Process new documents in batches
        for doc_batch in self._load_documents_in_batches(total_lines):
            if doc_batch:
                self.vectordb.add_documents(doc_batch)
                time.sleep(self.rate_limit)
                gc.collect()  # Help manage memory
        
        self.file_path = original_path
        print("Inside the add function and the persist will be called ::",self.persist_directory)
        if self.persist_directory:
            self.vectordb.persist()
            self.logger.info("Vector database persisted to disk")

    def ask(self, query: str) -> Dict[str, Any]:
        """Query the processed documents."""
        return self.genie.invoke(query)

    def persist(self) -> None:
        """Manually persist the vector database to disk if persistence is enabled."""
        print("I am invoked and the value of the directory is ",self.persist_directory)
        if self.persist_directory:
            print("Inside the persist function")
            self.vectordb.persist()
            self.logger.info("Vector database persisted to disk")


if __name__ == "__main__":
    
    embedding_function = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",  # e.g. ""
        openai_api_key="",
        azure_endpoint="",
        api_version="2023-05-15"                    
    )
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o-mini",  # e.g. "gpt-4"
        openai_api_key="",
        azure_endpoint="",
        api_version="2023-05-15"
    )


    genie=Genie('/Users/ganesh10/Documents/agent/CrewAi/trial1/fingate.jsonl',
                    embedding_function=embedding_function,
                  llm=llm,
                  persist_directory="./chroma_db")
    
    
    print('========')
    response =genie.ask("Give me the list of approved hotels I can stay at during campus visit , group it by cities.Give me name and address of the hotel ?")
    print(response)
    print('========')
