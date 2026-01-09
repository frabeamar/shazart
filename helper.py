import asyncio
import time
import uuid
from typing import AsyncGenerator

from dotenv import load_dotenv
from google.adk.agents import Agent, BaseAgent, LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, agent_tool, google_search
from google.adk.tools.tool_context import ToolContext
from google.genai import types

load_dotenv(".env")
# --- Define Tool Functions ---
# These functions simulate the actions of the specialist agents.
# running into 429 with gemini-2.0?
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL = "gemini-2.5-flash-lite"
# GEMINI_MODEL = "gemini-3-flash" NOT FOUND

def booking_handler(request: str) -> str:
    """
    Handles booking requests for flights and hotels.
    Args:
        request: The user's request for a booking.
    Returns:
        A confirmation message that the booking was handled.
    """
    print(
        "-------------------------- Booking Handler Called ----------------------------"
    )
    return f"Booking action for '{request}' has been simulated."


def info_handler(request: str) -> str:
    """
    Handles general information requests.
    Args:
        request: The user's question.
    Returns:
        A message indicating the information request was handled.
    """
    print("-------------------------- Info Handler Called ----------------------------")
    return (
        f"Information request for '{request}'. Result: Simulated information retrieval."
    )


def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    return f"Coordinator could not delegate request: '{request}'. Please clarify."


def run_agent(runner: Runner, request: str, user_id: str, session_id: str) -> list[str]:
    """
    Runs an agent pipeline and collects ALL text responses
    from all final responses across agents.
    """

    print(f"\n--- Running agent with request: '{request}' ---")

    responses: list[str] = []

    try:
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=request)],
            ),
        ):
            if not (event.is_final_response() and event.content):
                continue

            text = None

            if getattr(event.content, "text", None):
                text = event.content.text
            elif event.content.parts:
                text = (
                    event.author
                    + ":\n"
                    + "".join(part.text for part in event.content.parts if part.text)
                )

            if text:
                responses.append(text)

        print("Collected responses:")
        for r in responses:
            print("-", r)

        return responses

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


async def new_session(coordinator: BaseAgent):
    runner = InMemoryRunner(coordinator)
    user_id = "user_123"
    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    return runner, user_id, session_id


async def routing():
    """Main function to run the ADK example."""
    print("--- Google ADK Routing Example (ADK Auto-Flow Style) ---")
    print("Note: This requires Google ADK installed and authenticated.")

    # --- Create Tools from Functions ---
    booking_tool = FunctionTool(booking_handler)
    info_tool = FunctionTool(info_handler)

    # Define specialized sub-agents equipped with their respective tools
    booking_agent = Agent(
        name="Booker",
        model=GEMINI_MODEL,
        description="A specialized agent that handles all flight and hotel booking requests by calling the booking tool.",
        tools=[booking_tool],
    )

    info_agent = Agent(
        name="Info",
        model=GEMINI_MODEL,
        description="A specialized agent that provides general information and answers user questions by calling the info tool.",
        tools=[info_tool],
    )

    # Define the parent agent with explicit delegation instructions
    coordinator = Agent(
        name="Coordinator",
        model=GEMINI_MODEL,
        instruction=(
            "You are the main coordinator. Your only task is to analyze incoming user requests "
            "and delegate them to the appropriate specialist agent. Do not try to answer the user directly.\n"
            "- For any requests related to booking flights or hotels, delegate to the 'Booker' agent.\n"
            "- For all other general information questions, delegate to the 'Info' agent."
        ),
        description="A coordinator that routes user requests to the correct specialist agent.",
        # The presence of sub_agents enables LLM-driven delegation (Auto-Flow) by default.
        sub_agents=[booking_agent, info_agent],
    )

    runner, user_id, session_id = await new_session(coordinator)
    # Example Usage
    result_b = run_agent(
        runner, "What is the highest mountain in the world?", user_id, session_id
    )
    result_a = run_agent(runner, "Book me a hotel in Paris.", user_id, session_id)
    print(f"Final Output A: {result_a}")
    print(f"Final Output B: {result_b}")
    result_c = run_agent(
        runner, "Tell me a random fact.", user_id, session_id
    )  # Should go to Info
    print(f"Final Output C: {result_c}")
    result_d = run_agent(
        runner, "Find flights to Tokyo next month.", user_id, session_id
    )  # Should go to Booker
    print(f"Final Output D: {result_d}")


def mock_google_search(request: str) -> str:
    """
    Handles searching the net for information
    Args:
        request: The user's request for a booking.
    Returns:
        A confirmation message that the search was handled.
    """
    print(
        "-------------------------- Google Search Called ----------------------------"
    )
    return f"Search action for '{request}' has been simulated."


async def parallel_execution():
    # Part of agent.py --> Follow https://google.github.io/adk-docs/get-started/quickstart/ to learn the setup

    # Researcher 1: Renewable Energy
    researcher_agent_1 = LlmAgent(
        name="RenewableEnergyResearcher",
        model=GEMINI_MODEL,
        instruction="""You are an AI Research Assistant specializing in energy.
    Research the latest advancements in 'renewable energy sources'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches renewable energy sources.",
        tools=[mock_google_search],
        # Store result in state for the merger agent
        output_key="renewable_energy_result",
    )

    # Researcher 2: Electric Vehicles
    researcher_agent_2 = LlmAgent(
        name="EVResearcher",
        model=GEMINI_MODEL,
        instruction="""You are an AI Research Assistant specializing in transportation.
    Research the latest developments in 'electric vehicle technology'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches electric vehicle technology.",
        tools=[mock_google_search],
        # Store result in state for the merger agent
        output_key="ev_technology_result",
    )

    # Researcher 3: Carbon Capture
    researcher_agent_3 = LlmAgent(
        name="CarbonCaptureResearcher",
        model=GEMINI_MODEL,
        instruction="""You are an AI Research Assistant specializing in climate solutions.
    Research the current state of 'carbon capture methods'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches carbon capture methods.",
        tools=[mock_google_search],
        # Store result in state for the merger agent
        output_key="carbon_capture_result",
    )

    # --- 2. Create the ParallelAgent (Runs researchers concurrently) ---
    # This agent orchestrates the concurrent execution of the researchers.
    # It finishes once all researchers have completed and stored their results in state.
    parallel_research_agent = ParallelAgent(
        name="ParallelWebResearchAgent",
        sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
        description="Runs multiple research agents in parallel to gather information.",
    )

    # --- 3. Define the Merger Agent (Runs *after* the parallel agents) ---
    # This agent takes the results stored in the session state by the parallel agents
    # and synthesizes them into a single, structured response with attributions.
    merger_agent = LlmAgent(
        name="SynthesisAgent",
        model=GEMINI_MODEL,  # Or potentially a more powerful model if needed for synthesis
        instruction="""You are an AI Assistant responsible for combining research findings into a structured report.

    Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas. Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.

    **Crucially: Your entire response MUST be grounded *exclusively* on the information provided in the 'Input Summaries' below. Do NOT add any external knowledge, facts, or details not present in these specific summaries.**

    **Input Summaries:**

    *   **Renewable Energy:**
        {renewable_energy_result}

    *   **Electric Vehicles:**
        {ev_technology_result}

    *   **Carbon Capture:**
        {carbon_capture_result}

    **Output Format:**

    ## Summary of Recent Sustainable Technology Advancements

    ### Renewable Energy Findings
    (Based on RenewableEnergyResearcher's findings)
    [Synthesize and elaborate *only* on the renewable energy input summary provided above.]

    ### Electric Vehicle Findings
    (Based on EVResearcher's findings)
    [Synthesize and elaborate *only* on the EV input summary provided above.]

    ### Carbon Capture Findings
    (Based on CarbonCaptureResearcher's findings)
    [Synthesize and elaborate *only* on the carbon capture input summary provided above.]

    ### Overall Conclusion
    [Provide a brief (1-2 sentence) concluding statement that connects *only* the findings presented above.]

    Output *only* the structured report following this format. Do not include introductory or concluding phrases outside this structure, and strictly adhere to using only the provided input summary content.
    """,
        description="Combines research findings from parallel agents into a structured, cited report, strictly grounded on provided inputs.",
        # No tools needed for merging
        # No output_key needed here, as its direct response is the final output of the sequence
    )

    # --- 4. Create the SequentialAgent (Orchestrates the overall flow) ---
    # This is the main agent that will be run. It first executes the ParallelAgent
    # to populate the state, and then executes the MergerAgent to produce the final output.
    sequential_pipeline_agent = SequentialAgent(
        name="ResearchAndSynthesisPipeline",
        # Run parallel research first, then merge
        sub_agents=[parallel_research_agent, merger_agent],
        description="Coordinates parallel research and synthesizes the results.",
    )

    root_agent = sequential_pipeline_agent
    runner, user_id, session_id = await new_session(root_agent)
    run_agent(
        runner,
        "Conduct research on recent advancements in sustainable technologies and provide a structured report.",
        user_id,
        session_id,
    )


async def reflection():
    # The first agent generates the initial draft.
    generator = LlmAgent(
        name="DraftWriter",
        description="Generates initial draft content on a given subject.",
        instruction="Write a short, informative paragraph about the user's subject.",
        output_key="draft_text",  # The output is saved to this state key for intermediate processing.
        # it gets written out to event.actions.state_delta
        model=GEMINI_MODEL,
    )

    # The second agent critiques the draft from the first agent.
    reviewer = LlmAgent(
        name="FactChecker",
        description="Reviews a given text for factual accuracy and provides a structured critique.",
        instruction="""
        You are a meticulous fact-checker.
        1. Read the text provided in the state key 'draft_text'.
        2. Carefully verify the factual accuracy of all claims.
        3. Your final output must be a dictionary containing two keys:
        - "status": A string, either "ACCURATE" or "INACCURATE".
        - "reasoning": A string providing a clear explanation for your status, citing specific issues if any are found.
        """,
        output_key="review_output",  # The structured dictionary is saved here.
        model=GEMINI_MODEL,
    )

    # The SequentialAgent ensures the generator runs before the reviewer.
    review_pipeline = SequentialAgent(
        name="WriteAndReview_Pipeline", sub_agents=[generator, reviewer]
    )
    runner, user_id, session_id = await new_session(review_pipeline)
    run_agent(
        runner,
        "The impact of climate change on polar bear populations.",
        user_id=user_id,
        session_id=session_id,
    )

    # Execution Flow:
    # 1. generator runs -> saves its paragraph to state['draft_text'].
    # 2. reviewer runs -> reads state['draft_text'] and saves its dictionary output to state['review_output'].
    # the output of the reviewer is a code box with a json inside
    # should it be parsed? with pydantic?


async def builtin_google_search():
    # Define Agent with access to search tool
    root_agent = Agent(
        name="basic_search_agent",
        # model="gemini-2.0-flash-exp",
        model=GEMINI_MODEL,
        description="Agent to answer questions using Google Search.",
        instruction="I can answer your questions by searching the internet. Just ask me anything!",
        tools=[
            google_search
        ],  # Google Search is a pre-built tool to perform Google searches.
    )
    runner, user_id, session_id = await new_session(root_agent)
    run_agent(
        runner,
        "What are the latest advancements in AI technology as of 2024?",
        user_id,
        session_id,
    )


async def call_agent_async(query):
    code_agent = LlmAgent(
        name="calculator_agent",
        model=GEMINI_MODEL,
        code_executor=BuiltInCodeExecutor(),
        instruction="""You are a calculator agent.
   When given a mathematical expression, write and execute Python code to calculate the result.
   Return only the final numerical result as plain text, without markdown or code blocks.
   """,
        description="Executes Python code to perform calculations.",
    )
    # Session and Runner
    session_id = str(uuid.uuid4())
    user_id = "user_456"
    app_name = "calculator"
    session_service = InMemorySessionService()  # temporary memory of the session
    # stored in RAM, gets deleted after program end
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    runner = Runner(  # handles cloud execution -> not sure for which aspects it is different than InMemoryRunner
        agent=code_agent,
        app_name=app_name,
        # user_id=user_id,
        # session_id=session_id,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=query)])
    print(f"\n--- Running Query: {query} ---")
    final_response_text = "No final text response captured."
    try:
        # Use run_async; need async for to loop over a generator of async events
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            print(f"Event ID: {event.id}, Author: {event.author}")

            # --- Check for specific parts FIRST ---
            # has_specific_part = False
            if event.content and event.content.parts and event.is_final_response():
                for part in event.content.parts:  # Iterate through all parts
                    breakpoint()
                    if part.executable_code:
                        # Access the actual code string via .code
                        print(
                            f"  Debug: Agent generated code:\n```python\n{part.executable_code.code}\n```"
                        )
                    elif part.code_execution_result:
                        # Access outcome and output correctly
                        print(
                            f"  Debug: Code Execution Result: {part.code_execution_result.outcome} - Output:\n{part.code_execution_result.output}"
                        )
                    # Also print any text parts found in any event for debugging
                    elif part.text and not part.text.isspace():
                        print(f"  Text: '{part.text.strip()}'")

                # --- Check for final response AFTER specific parts ---
                text_parts = [part.text for part in event.content.parts if part.text]
                final_result = "".join(text_parts)
                print(f"==> Final Agent Response: {final_result}")

    except Exception as e:
        print(f"ERROR during agent run: {e}")


class TaskExecutor(BaseAgent):
    """A specialized agent with custom, non-LLM behavior."""

    name: str = "TaskExecutor"
    description: str = "Executes a predefined task."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[
        Event, None
    ]:  # AsyncGenerator[yield_type, send_type] you can send in values with send()!
        """Custom implementation logic for the task."""
        print("I am executing the task...", self.name)
        content = types.Content(role="system", parts=[types.Part(text="Task fineshed")])
        yield Event(author=self.name, content=content)


async def multi_agent_collaboration():
    # Correctly implement a custom agent by extending BaseAgent
    # Define individual agents with proper initialization
    # LlmAgent requires a model to be specified.
    greeter = LlmAgent(
        name="Greeter",
        model=GEMINI_MODEL,
        instruction="You are a friendly greeter.",
    )
    task_doer = TaskExecutor()  # Instantiate our concrete custom agent

    # Create a parent agent and assign its sub-agents
    # The parent agent's description and instructions should guide its delegation logic.
    coordinator = LlmAgent(
        name="Coordinator",
        model=GEMINI_MODEL,
        description="A coordinator that can greet users and execute tasks.",
        instruction="When asked to greet, delegate to the Greeter. When asked to perform a task, delegate to the TaskExecutor.",
        sub_agents=[greeter, task_doer],
    )

    # The ADK framework automatically establishes the parent-child relationships.
    # These assertions will pass if checked after initialization.
    assert greeter.parent_agent == coordinator
    assert task_doer.parent_agent == coordinator
    runner, user_id, session_id = await new_session(coordinator)
    run_agent(
        runner,
        "Please greet me and then perform the task.",
        user_id,
        session_id,
    )


class ConditionChecker(BaseAgent):
    """A custom agent that checks for a 'completed' status in the session state."""

    name: str = "ConditionChecker"
    description: str = "Checks if a process is complete and signals the loop to stop."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Checks state and yields an event to either continue or stop the loop."""
        status = context.session.state.get("status", "pending")
        is_done = status == "completed"

        if is_done:
            # Escalate to terminate the loop when the condition is met.
            # escalete -> gives to the parent agent
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            # Yield a simple event to continue the loop.
            content = types.Content(
                role="system",
                parts=[types.Part(text="Condition not met, continuing loop.")],
            )
            yield Event(author=self.name, content=content)


async def multi_agent_loop():
    process_step = LlmAgent(
        name="ProcessingStep",
        model=GEMINI_MODEL,
        instruction="You are a step in a longer process. Perform your task. If you are the final step, update session state by setting 'status' to 'completed'.",
    )

    # The LoopAgent orchestrates the workflow.
    poller = LoopAgent(
        name="StatusPoller",
        max_iterations=10,
        sub_agents=[
            process_step,
            ConditionChecker(),
        ],
    )
    runner, user_id, session_id = await new_session(poller)
    run_agent(
        runner,
        "Execute the multi-step process until completion.",
        user_id,
        session_id,
    )


def generate_image(prompt: str) -> dict:
    """
    Generates an image based on a textual prompt.

    Args:
        prompt: A detailed description of the image to generate.

    Returns:
        A dictionary with the status and the generated image bytes.
    """
    breakpoint()
    print(f"TOOL: Generating image for prompt: '{prompt}'")
    # In a real implementation, this would call an image generation API.
    # For this example, we return mock image data.
    mock_image_bytes = b"mock_image_data_for_a_cat_wearing_a_hat"
    return {
        "status": "success",
        # The tool returns the raw bytes, the agent will handle the Part creation.
        "image_bytes": mock_image_bytes,
        "mime_type": "image/png",
    }


async def agent_as_tools():
    image_generator_agent = LlmAgent(
        name="ImageGen",
        model=GEMINI_MODEL,
        description="Generates an image based on a detailed text prompt.",
        instruction=(
            "You are an image generation specialist. Your task is to take the user's request "
            "and use the `generate_image` tool to create the image. "
            "The user's entire request should be used as the 'prompt' argument for the tool. "
            "After the tool returns the image bytes, you MUST output the image."
        ),
        tools=[generate_image],
    )

    # 3. Wrap the agent in an AgentTool.
    # The description here is what the parent agent sees.
    image_tool = agent_tool.AgentTool(
        agent=image_generator_agent,
    )

    # 4. The parent agent remains unchanged. Its logic was correct.
    artist_agent = LlmAgent(
        name="Artist",
        model=GEMINI_MODEL,
        instruction=(
            "You are a creative artist. First, invent a creative and descriptive prompt for an image. "
            "Then, use the `ImageGen` tool to generate the image using your prompt."
        ),
        tools=[image_tool],
    )
    runner, user_id, session_id = await new_session(artist_agent)
    run_agent(
        runner,
        "Create an image of a cat wearing a hat.",
        user_id,
        session_id,
    )


# --- Define the Recommended Tool-Based Approach ---
def log_user_login(tool_context: ToolContext) -> dict:
    """
    Updates the session state upon a user login event.
    This tool encapsulates all state changes related to a user login.
    Args:
        tool_context: Automatically provided by ADK, gives access to session state.
    Returns:
        A dictionary confirming the action was successful.
    """
    # Access the state directly through the provided context.
    state = tool_context.state

    # Get current values or defaults, then update the state.
    # This is much cleaner and co-locates the logic.
    breakpoint()
    login_count = state.get("user:login_count", 0) + 1
    state["user:login_count"] = login_count
    state["task_status"] = "active"
    state["user:last_login_ts"] = time.time()
    state["temp:validation_needed"] = True

    # print("State updated from within the `log_user_login` tool.")
    print("STATE INSIDE TOOL:", tool_context.state)

    return {
        "status": "success",
        "message": f"User login tracked. Total logins: {login_count}.",
    }


async def session_with_memory():
    session_service = InMemorySessionService()
    app_name, user_id, session_id = "state_app_tool", "user3", "session3"
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"user:login_count": 0, "task_status": "idle"},  # add initial state
    )
    print(f"Initial state: {session.state}")

    # 2. Simulate a tool call (in a real app, the ADK Runner does this)
    # We create a ToolContext manually just for this standalone example.
    agent = Agent(
        name="LoginAgent",
        model=GEMINI_MODEL,
        description="You are responsible for loggin in the user and updating session state.",
        instruction="Log the user using the log_user_login tool.",
        tools=[log_user_login],
    )

    runner = Runner(app_name=app_name, agent=agent, session_service=session_service)
    run_agent(
        runner,
        "Log in the user and update session state.",
        user_id,
        session_id,
    )
    print("STATE AFTER TOOL CALL:", session.state)
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if session:
        print(session.state)


async def main():
    # await builtin_google_search()
    # await call_agent_async("Calculate the value of (5 + 7) * 3")
    # await multi_agent_collaboration()
    # await multi_agent_loop() # very expensive to run
    # await agent_as_tools()
    await session_with_memory()


if __name__ == "__main__":
    asyncio.run(main())
