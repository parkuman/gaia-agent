from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader, YoutubeLoader
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START
from openrouter import ChatOpenRouter
import os
import re
import math
from typing import Optional
from urllib.parse import urlparse
import uuid
import requests
import tempfile
from dotenv import load_dotenv
import whisper
from pydub import AudioSegment

load_dotenv()


@tool
def transcribe_audio_file_tool(filepath: str) -> str:
    """
    Read an audio file and transcribe it using Whisper.
    Args:
        filepath (str): The path to the audio file (supports mp3, wav, etc.)
    Returns:
        str: The transcribed text from the audio file
    """
    try:
        # Load the Whisper model (will download on first use)
        model = whisper.load_model("tiny")

        # Convert audio to wav if it's not already (Whisper expects PCM WAV)
        audio = AudioSegment.from_file(filepath)

        # Create a temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(temp_wav.name, format="wav")

        # Transcribe the audio
        result = model.transcribe(temp_wav.name)

        # Clean up temporary file
        os.unlink(temp_wav.name)

        return result["text"]
    except Exception as e:
        return f"Error transcribing audio file: {e}"


@tool
def youtube_transcript_tool(youtube_video_url: str) -> str:
    """
    Returns a text transcript of a provided youtube video URL.
    Args:
        youtube_video_urL (urL): The url of the video you want the transcript for.
    Returns:
        The transcript of the youtube video as text
    """
    youtube_transcript = YoutubeLoader.from_youtube_url(
        youtube_url=youtube_video_url).load()
    return youtube_transcript


@tool
def read_text_file_tool(filepath: str) -> str:
    """
    Read the content of a file and return it as a string.
    Args:
        filepath (str): The path to the file to read.
    Returns:
        str: The content of the file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_content_to_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."


@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"


@tool
def wikipedia_search_tool(query: str) -> str:
    """Query Wikipedia and return a max of 2 results.
    Args:
        query: The search query.
    Returns: 
        the wikipedia documents most relevant to the query
    """
    wiki_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{
                doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in wiki_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}


@tool
def web_search_tool(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(input=query)
    return {"web_results": search_docs}


@tool
def arxiv_search_tool(query: str) -> str:
    """Query Arxiv and return a max of 3 results.
    Args:
        query: The search query.
    """
    arxiv_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{
                doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in arxiv_docs
        ]
    )
    return {"arxiv_results": formatted_search_docs}


@tool
def multiply(a: float, b: float) -> float:
    """multiplies two numbers and returns the result

    Args:
     a: first number
     b: second number
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """divides two numbers and returns the result

    Args:
     a: first number
     b: second number
    """
    return a / b


@tool
def add(a: float, b: float) -> float:
    """adds two numbers and returns the result

    Args:
     a: first number
     b: second number
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """subtracts b from a and returns the result

    Args:
     a: first number
     b: second number
    """
    return a - b


@tool
def modulo(a: float, b: float) -> float:
    """returns the result of a modulo b

    Args:
     a: first number
     b: second number
    """
    return a % b


@tool
def power(a: float, b: float) -> float:
    """
    Get the power of two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a**b


@tool
def square_root(a: float) -> float | complex:
    """
    Get the square root of a number.
    Args:
        a (float): the number to get the square root of
    """
    if a >= 0:
        return a**0.5
    return math.sqrt(a)


class Agent:
    """A customizable AI agent that can handle various tasks."""

    def __init__(self, model_name: str = "google/gemini-2.0-flash-exp:free"):
        """Initialize the agent with a specified model.

        Args:
            model_name (str): The name of the LLM model to use
        """

        self.tools = [
            wikipedia_search_tool,
            arxiv_search_tool,
            web_search_tool,
            download_file_from_url,
            write_content_to_file,
            read_text_file_tool,
            youtube_transcript_tool,
            transcribe_audio_file_tool,
            add,
            subtract,
            multiply,
            divide,
            modulo,
            power,
            square_root,
        ]
        self.llm = ChatOpenRouter(model_name=model_name)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.system_prompt = SystemMessage(
            content="""You are a general AI agent with the ability to call tools depending on the task. I will ask you a question.

Please attempt to answer this question by calling any of the provided tools. Once you feel you have a sufficient answer, 
finish your answer by simply outputting the exact answer, nothing more, nothing less. 

YOUR FINAL ANSWER should be:
- A number OR
- As few words as possible OR
- A comma separated list of numbers and/or strings
- If you feel you don't have the right tools for the job / don't know the answer, please output [TOOL_NEEDED] and indicate what tool you think might be useful to complete that task.

Guidelines:
- If asked for a number: Don't use commas or units (like $ or %) unless specified
- If asked for a string: Don't use articles or abbreviations (e.g. for cities), and write digits in plain text
- If asked for a comma separated list: Apply the above rules to each element""")

        builder = StateGraph(MessagesState)
        memory = MemorySaver()

        # NODES
        builder.add_node("assistant", self._assistant_node)
        builder.add_node("tools", ToolNode(self.tools))

        # EDGES
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        self.graph = builder.compile(checkpointer=memory)

    def _sanitize_filename(self, question: str) -> str:
        """Sanitize the question to be used as a filename.

        Args:
            question (str): The question to sanitize

        Returns:
            str: A sanitized version of the question suitable for use as a filename
        """
        # Replace non-alphanumeric characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', question)
        # Truncate if too long (max 100 chars)
        sanitized = sanitized[:100]
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed_question"
        return sanitized

    def __call__(self, question: str) -> str:
        """Process a question and return an answer.

        Args:
            question (str): The question or task to process

        Returns:
            str: The agent's response
        """

        # Create an identifier from the question
        question_identifier = self._sanitize_filename(question)
        config = {"configurable": {"thread_id": question_identifier}}

        # Create the messages from the prompt template
        question_message = HumanMessage(content=question)

        result = self.graph.invoke(
            {"messages": [self.system_prompt, question_message]}, config)

        # prepare a filename where we're going to save the output for debugging later
        filename = question_identifier + ".txt"
        filepath = os.path.join(os.getcwd(), "output", filename)

        # Remove the file if it already exists
        if os.path.exists(filepath):
            os.remove(filepath)

        # Save the pretty-printed messages to the file
        with open(filepath, 'w') as f:
            for m in result['messages']:
                # Capture the pretty-printed output
                import io
                from contextlib import redirect_stdout

                output = io.StringIO()
                with redirect_stdout(output):
                    m.pretty_print()

                f.write(output.getvalue())
                f.write("\n")

        # Also print to console as before
        for m in result['messages']:
            m.pretty_print()

        messages = self.graph.get_state(config).values["messages"]
        final_answer = messages[-1].content

        return final_answer

    def _assistant_node(self, state: MessagesState):
        return {"messages": [self.llm_with_tools.invoke([self.system_prompt] + state["messages"])]}
