import os
import re

from openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode


def multiply(a: float, b: float) -> float:
    """multiplies two numbers and returns the result

    Args:
     a: first number
     b: second number
    """
    return a * b


def divide(a: float, b: float) -> float:
    """divides two numbers and returns the result

    Args:
     a: first number
     b: second number
    """
    return a / b


def add(a: float, b: float) -> float:
    """adds two numbers and returns the result

    Args:
     a: first number
     b: second number
    """
    return a + b


def subtract(a: float, b: float) -> float:
    """subtracts b from a and returns the result

    Args:
     a: first number
     b: second number
    """
    return a - b


def modulo(a: float, b: float) -> float:
    """returns the result of a modulo b

    Args:
     a: first number
     b: second number
    """
    return a % b


class Agent:
    """A customizable AI agent that can handle various tasks."""

    def __init__(self, model_name: str = "google/gemini-2.5-pro-preview"):
        """Initialize the agent with a specified model.

        Args:
            model_name (str): The name of the LLM model to use
        """

        self.tools = [add, subtract, multiply, divide, modulo]
        self.llm = ChatOpenRouter(model_name=model_name)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.system_prompt = SystemMessage(
            content="""You are a general AI agent with the ability to call tools depending on the task. I will ask you a question.

Please attempt to answer this question by calling any of the provided tools. Once you feel you have a sufficient answer, 
finish your answer by simply outputting the exact answer, nothing more, nothing less. If you feel you don't have the right tools for the job, please
output [TOOL_NEEDED] and indicate what tool you think might be useful to complete that task.

YOUR FINAL ANSWER should be:
- A number OR
- As few words as possible OR
- A comma separated list of numbers and/or strings

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
