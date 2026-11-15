---
description: Build the trace evalproof's checks read from Google ADK, LangChain or LangGraph, the Claude Agent SDK, OpenAI Agents, LlamaIndex, CrewAI, Pydantic AI, DSPy, Haystack, AutoGen, Semantic Kernel, OpenTelemetry spans, plain Python or MCP.
---

# Agent SDK adapters

An adapter turns what an agent SDK returns into the trace that the `checks` scorer reads. This page is for anyone testing an agent built on Google ADK, LangChain or LangGraph, the Claude Agent SDK, the OpenAI Agents SDK, LlamaIndex, CrewAI, Pydantic AI, DSPy, Haystack, AutoGen, Semantic Kernel, plain Python or another language.

A python system (`target: python:agent:run`) returns `{"output", "trace", ...}`, and the `trace`
is what the `checks` scorer reads: the tools called, their arguments and results, the rounds,
the tokens. `evalproof.adapters` builds that dict from what eleven agent SDKs return. The
adapters import no SDK and read objects or plain dicts, so they work with any version that keeps
the field names they cite; each module's header gives the package version and file:line of every
field it reads. Tokens and cost roll up from the trace, so `max_cost_usd` and the budgets see an
agent's spend. Cost is set only where the SDK reports one (Claude); otherwise the target's
`pricing` applies. A turn boundary is read from the input where it shows (ADK user events, a
LangChain human message, a Claude `ResultMessage`), else every call is turn 0. For a multi-turn
case where each user turn is a separate call to the SDK, pass one item per turn: a list of ADK
event lists, a list of Claude message lists (one per `query()` call), or a list of OpenAI
results. The python function gets the later turns as `ctx.turns`.

An agent that already exports OpenTelemetry spans needs no SDK adapter: `otel.from_spans` reads
them (experimental, see the OpenTelemetry tab below).

!!! warning "Experimental adapters"

    The LlamaIndex, CrewAI, Pydantic AI, DSPy, Haystack, AutoGen and Semantic Kernel adapters are
    experimental. Their field names were read from each SDK's source and are tested with fake
    objects of those shapes, and they have not yet been run against a live agent of each
    framework. Each one raises one `UserWarning` the first time it is called in a process. It is a
    warning and not a line on stderr, because the adapter runs inside your own python target:
    filter it with `warnings.filterwarnings`, or turn it into an error with `-W error`. Each tab
    lists what the adapter leaves out.

## Choose your SDK

=== "Google ADK"

    `adk.from_events(events)`; pairs `function_call` and `function_response` parts
    by id, then by name and order. A tool that returns `{"error": ...}` is recorded as failed.
    `run_async` does not yield the user's message by default, so for several turns pass one event
    list per run (or `session.events`).

    ```python title="agent.py"
    import asyncio

    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from evalproof.adapters import adk


    def get_weather(city: str) -> dict:
        """Current weather for a city."""
        return {"city": city, "temp_c": 20}


    agent = LlmAgent(name="assistant", model="gemini-2.5-flash", tools=[get_weather])
    runner = InMemoryRunner(agent=agent, app_name="evals")


    async def _events(question: str) -> list:
        session = await runner.session_service.create_session(app_name="evals", user_id="eval")
        message = types.Content(role="user", parts=[types.Part.from_text(text=question)])
        return [
            e
            async for e in runner.run_async(user_id="eval", session_id=session.id, new_message=message)
        ]


    def run(question: str) -> dict:
        return adk.from_events(asyncio.run(_events(question)))
    ```

=== "LangChain and LangGraph"

    `langchain.from_messages(messages)` takes the message list or the
    state dict an agent returns; `ToolMessage.status == "error"` marks a failed call.

    ```python title="agent.py"
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    from evalproof.adapters import langchain


    @tool
    def next_meeting(day: str) -> str:
        """The first calendar entry on a day."""
        return f"{day}: 10:00 planning"


    agent = create_agent("openai:gpt-5.5", [next_meeting], system_prompt="Answer calendar questions.")


    def run(question: str) -> dict:
        state = agent.invoke({"messages": [{"role": "user", "content": question}]})
        return langchain.from_messages(state)
    ```

=== "Claude Agent SDK"

    `claude_agent_sdk.from_messages(messages)`; `ToolResultBlock.is_error`
    marks a failed call, the usage comes from the per-model `model_usage` of the final
    `ResultMessage`, and `cost_usd` is its `total_cost_usd`. Those are running totals of a session;
    pass one message list per `query()` call and the sessions' totals are added.

    ```python title="agent.py"
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query, tool

    from evalproof.adapters import claude_agent_sdk


    @tool("get_weather", "Current weather for a city", {"city": str})
    async def get_weather(args):
        return {"content": [{"type": "text", "text": f"{args['city']}: 20 C, clear"}]}


    options = ClaudeAgentOptions(
        mcp_servers={"weather": create_sdk_mcp_server(name="weather", tools=[get_weather])},
        system_prompt="Answer weather questions.",
        max_turns=5,  # also allow the tool with allowed_tools, as the SDK docs show
    )


    async def _messages(question: str) -> list:
        return [m async for m in query(prompt=question, options=options)]


    def run(question: str) -> dict:
        return claude_agent_sdk.from_messages(asyncio.run(_messages(question)))
    ```

=== "OpenAI Agents SDK"

    `openai_agents.from_result(result)` takes a `RunResult`, or a list of
    them for a multi-turn case (one result per user turn). The SDK has no field that says a tool
    failed: it turns an exception into a plain string, by default the fixed text in
    `openai_agents.DEFAULT_TOOL_ERROR`, and an output equal to that text is recorded as failed. For a
    tool with its own `failure_error_function`, pass `error_if=lambda output: ...`, which replaces
    the default test (`error_if=lambda output: False` turns it off).

    ```python title="agent.py"
    from agents import Agent, Runner, function_tool

    from evalproof.adapters import openai_agents


    @function_tool
    def next_meeting(day: str) -> str:
        """The first calendar entry on a day."""
        return f"{day}: 10:00 planning"


    agent = Agent(name="assistant", instructions="Answer calendar questions.", tools=[next_meeting])


    def run(question: str) -> dict:
        return openai_agents.from_result(Runner.run_sync(agent, question))
    ```

=== "LlamaIndex (experimental)"

    `llama_index.from_events(events)` takes the events `handler.stream_events()` yields for an
    agent workflow (`FunctionAgent`, `ReActAgent`), a list of event lists for a multi-turn case
    (one per `agent.run`), or the `AgentChatResponse` of an older agent or chat engine. A call is
    counted once although its `ToolCall` event and the `AgentOutput` that made it both show it,
    and a `ToolOutput` with `is_error` marks a failed call. Left out: tokens and cost.

    ```python title="agent.py"
    import asyncio

    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.llms.openai import OpenAI

    from evalproof.adapters import llama_index


    def get_weather(city: str) -> str:
        """Current weather for a city."""
        return f"{city}: 20 C, clear"


    agent = FunctionAgent(
        tools=[get_weather], llm=OpenAI(model="gpt-5.5"), system_prompt="Answer weather questions."
    )


    async def _events(question: str) -> list:
        handler = agent.run(question)
        events = [event async for event in handler.stream_events()]
        await handler
        return events


    def run(question: str) -> dict:
        return llama_index.from_events(asyncio.run(_events(question)))
    ```

=== "CrewAI (experimental)"

    `crewai.from_output(output)` takes the `CrewOutput` that `crew.kickoff(...)` returns (or a
    list of them, one per turn). The calls come from each task's `messages`, the
    failed ones from its `tool_failures`, and the tokens from the crew's `token_usage`. Left out:
    agents whose model does not use native tool calling (their calls are only text in the
    messages), the model name, and cost.

    ```python title="agent.py"
    from crewai import Agent, Crew, Task
    from crewai.tools import tool

    from evalproof.adapters import crewai


    @tool("get_weather")
    def get_weather(city: str) -> str:
        """Current weather for a city."""
        return f"{city}: 20 C, clear"


    analyst = Agent(
        role="Weather analyst",
        goal="Answer weather questions",
        backstory="You answer from the weather tool.",
        tools=[get_weather],
    )
    task = Task(description="{question}", expected_output="A one-line answer", agent=analyst)
    crew = Crew(agents=[analyst], tasks=[task])


    def run(question: str) -> dict:
        return crewai.from_output(crew.kickoff(inputs={"question": question}))
    ```

=== "Pydantic AI (experimental)"

    `pydantic_ai.from_messages(result)` takes the run result (or the list `all_messages()`
    returns, or the dicts Pydantic AI's own serialisation produces). A tool return whose
    `outcome` is not `"success"`, or a retry prompt that names a tool, marks a failed call. An
    agent with structured output also shows its output tool as a call. Left out: tool calls
    that the model provider ran itself, and cost.

    ```python title="agent.py"
    from pydantic_ai import Agent

    from evalproof.adapters import pydantic_ai

    agent = Agent("openai:gpt-5.5", system_prompt="Answer weather questions.")


    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Current weather for a city."""
        return f"{city}: 20 C, clear"


    def run(question: str) -> dict:
        return pydantic_ai.from_messages(agent.run_sync(question))
    ```

=== "DSPy (experimental)"

    `dspy_adapter.from_prediction(prediction)` reads the `dspy.ReAct` prediction: the `history` of
    DSPy 3.5 and the `trajectory` of 3.4 and earlier. The reserved `submit` and `finish` steps are
    not calls. A tool that raised is a failed call. The tokens come from `get_lm_usage()`, which
    is filled when usage tracking is on (`track_usage=True`). The reply is the one output field,
    else all of them as JSON; pass `output_field="answer"` to pick one. Left out: cached tokens
    and cost. The module is imported under another name because `dspy` is also the SDK.

    ```python title="agent.py"
    import dspy

    from evalproof.adapters import dspy as dspy_adapter

    dspy.configure(lm=dspy.LM("openai/gpt-5.5"), track_usage=True)


    def get_weather(city: str) -> str:
        """Current weather for a city."""
        return f"{city}: 20 C, clear"


    react = dspy.ReAct("question -> answer", tools=[get_weather])


    def run(question: str) -> dict:
        return dspy_adapter.from_prediction(react(question=question), output_field="answer")
    ```

=== "Haystack (experimental)"

    `haystack.from_messages(messages)` takes the list of `ChatMessage` objects, or the dict
    `Agent.run` returns (its `messages`). A `ToolCallResult` with `error` set marks a failed call,
    and the tokens come from each reply's `meta["usage"]`. Left out: cached tokens, cost, and an
    `Agent` run read only through `last_message` (pass `messages`).

    ```python title="agent.py"
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Tool

    from evalproof.adapters import haystack


    def get_weather(city: str) -> str:
        return f"{city}: 20 C, clear"


    weather = Tool(
        name="get_weather",
        description="Current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        function=get_weather,
    )
    agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-5.5"), tools=[weather])
    agent.warm_up()


    def run(question: str) -> dict:
        return haystack.from_messages(agent.run(messages=[ChatMessage.from_user(question)]))
    ```

=== "AutoGen (experimental)"

    `autogen.from_messages(result)` takes the `TaskResult` that `agent.run(...)` or a team's
    `run` returns (or its `messages`). Calls come from the `ToolCallRequestEvent` and
    `ToolCallExecutionEvent` messages, and a result with `is_error` marks a failed call. The
    usage has no model name, because AutoGen's `RequestUsage` carries none. Left out:
    cached tokens and cost.

    ```python title="agent.py"
    import asyncio

    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    from evalproof.adapters import autogen


    async def get_weather(city: str) -> str:
        """Current weather for a city."""
        return f"{city}: 20 C, clear"


    async def _run(question: str):
        client = OpenAIChatCompletionClient(model="gpt-5.5")
        agent = AssistantAgent(
            "assistant", model_client=client, tools=[get_weather], reflect_on_tool_use=True
        )
        try:
            return await agent.run(task=question)
        finally:
            await client.close()


    def run(question: str) -> dict:
        return autogen.from_messages(asyncio.run(_run(question)))
    ```

=== "Semantic Kernel (experimental)"

    `semantic_kernel.from_history(history)` takes the `ChatHistory` a function-calling run fills
    (or its `messages`). A `FunctionCallContent` is a call, named as the SDK names it
    (`<plugin>-<function>`, so a check says `weather-get_weather`), and a `FunctionResultContent`
    answers it by id. Left out because the SDK shows no field for them: failed calls (every
    result counts as succeeded) and token usage.

    ```python title="agent.py"
    import asyncio

    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai import FunctionChoiceBehavior
    from semantic_kernel.connectors.ai.open_ai import (
        OpenAIChatCompletion,
        OpenAIChatPromptExecutionSettings,
    )
    from semantic_kernel.contents import ChatHistory
    from semantic_kernel.functions import kernel_function

    from evalproof.adapters import semantic_kernel


    class Weather:
        @kernel_function(name="get_weather", description="Current weather for a city")
        def get_weather(self, city: str) -> str:
            return f"{city}: 20 C, clear"


    service = OpenAIChatCompletion(ai_model_id="gpt-5.5")
    kernel = Kernel()
    kernel.add_service(service)
    kernel.add_plugin(Weather(), plugin_name="weather")
    settings = OpenAIChatPromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto())


    def run(question: str) -> dict:
        history = ChatHistory()
        history.add_user_message(question)
        asyncio.run(service.get_chat_message_content(history, settings, kernel=kernel))
        return semantic_kernel.from_history(history)
    ```

=== "Plain Python or MCP"

    `TraceRecorder` builds the same dict by hand, and `mcp_call` records
    an MCP `CallToolResult` (`isError` makes the call fail):

    ```python title="agent.py"
    import asyncio

    from evalproof.adapters import TraceRecorder, mcp_call


    def run(question: str) -> dict:
        rec = TraceRecorder()
        rec.round()
        rec.tool("get_weather", {"city": "Paris"}, result={"temp_c": 20})
        args = {"day": "Monday"}
        result = asyncio.run(mcp_session.call_tool("next_meeting", args))  # an mcp ClientSession
        mcp_call(rec, "next_meeting", args, result)
        rec.usage(model="m", tokens_in=120, tokens_out=30)
        rec.reply("It is 20 degrees in Paris, and the first meeting is at 10:00.")
        return rec.result()
    ```

=== "TypeScript (Vercel AI SDK)"

    An app in another language joins through an `http` target
    (`target: http://localhost:3000/eval`). evalproof POSTs `{"id", "input"}` and reads
    `{"output", "trace", ...}` back, with the `trace` in evalproof's field names, so the app maps
    `generateText`'s `text`, `toolCalls`, `toolResults` and `usage` (all steps, in `ai` 7) itself:

    ```ts title="route.ts"
    import { generateText } from "ai";

    export async function POST(req: Request) {
      const { input } = await req.json();
      const r = await generateText({ model, tools, prompt: input }); // your model and tools
      return Response.json({
        output: r.text,
        trace: {
          tool_calls: r.toolCalls.map((c) => ({
            id: c.toolCallId,
            name: c.toolName,
            args: c.input,
            result: r.toolResults.find((t) => t.toolCallId === c.toolCallId)?.output,
          })),
          usage: [{ role: "agent", tokens_in: r.usage.inputTokens ?? 0, tokens_out: r.usage.outputTokens ?? 0 }],
        },
      });
    }
    ```

    Set `ok: false` and `error` on a call that failed, and `turn` (counting from 0) on the calls
    of a multi-turn case. `rounds` is the number of model calls (`r.steps.length`), `responses` one
    reply per turn.

=== "OpenTelemetry spans (experimental)"

    !!! warning "Experimental"

        `otel.from_spans` reads the OpenTelemetry GenAI conventions, which are still in
        Development status, and it was built from the specifications and not from spans exported
        by each framework. The first call in a process raises one `UserWarning` saying so. It is a
        warning and not a line on stderr, because the adapter runs inside your own python target:
        filter it with `warnings.filterwarnings`, or turn it into an error with `-W error`.

    If the agent already exports spans, `from_spans` builds the trace from them and no SDK
    objects are needed. It takes a path, an OTLP/JSON object (`resourceSpans`), one span or a
    list of spans, as OTLP spans or as the Python SDK's `span.to_json()` objects:

    ```python title="agent.py"
    from evalproof.adapters import otel


    def run(question: str) -> dict:
        run_agent(question)  # your agent, exporting spans to ./spans.jsonl
        return otel.from_spans("spans.jsonl")
    ```

    A file can hold one OTLP object, one per line (the collector's `file` exporter with
    `format: json` and no compression), or several pretty-printed objects in a row (what
    `ConsoleSpanExporter` prints). The collector's `proto` and zstd forms are binary and are not
    read. When the spans belong to more than one trace, pass `trace_id="..."` (hex, with or
    without `0x`); without it `from_spans` raises `ValueError` and lists the ids it found. Spans
    are ordered by start time.

    | Convention | Tool span | Name, arguments, result, id |
    |---|---|---|
    | GenAI semantic conventions, Google ADK schema 2, Vercel `OpenTelemetry` | `gen_ai.operation.name` is `execute_tool` | `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `gen_ai.tool.call.id` (ADK's `<not specified>` is no id) |
    | Google ADK schema 1 | the same, with `gcp.vertex.agent.*` | `gen_ai.tool.name`, `gcp.vertex.agent.tool_call_args`, `gcp.vertex.agent.tool_response`, `gen_ai.tool.call.id` |
    | OpenInference | `openinference.span.kind` is `TOOL` | `tool.name`, `input.value`, `output.value`, `tool.id` |
    | Vercel `LegacyOpenTelemetry` | `ai.toolCall` | `ai.toolCall.name`, `ai.toolCall.args`, `ai.toolCall.result`, `ai.toolCall.id` |

    When tools run in parallel, ADK writes one more `execute_tool (merged)` span with the tool
    name `(merged tools)` beside the span of each call. It is not a call and is skipped.

    An `error.type` attribute or an ERROR status makes the call fail (`ok: false`, with the type
    and the status message as the error). Arguments that are not a JSON object are kept as
    `{"input": ...}`, and a result that is JSON text is decoded. `rounds` counts the model-call
    spans (`chat`, `generate_content`, `text_completion`, OpenInference `LLM`, Vercel
    `doGenerate` and `doStream`, ADK schema 1 `call_llm`). Tokens are summed per model from
    `gen_ai.usage.input_tokens` and `output_tokens` (or the deprecated `prompt_tokens` and
    `completion_tokens`), `llm.token_count.prompt` and `completion`, and
    `ai.usage.promptTokens` and `completionTokens`. The reply is the text of the last model-call
    span.

    **Content capture is off by default in most of these, and a check on arguments, results or
    the reply cannot pass without it.** A tool span with no arguments and no result (or the
    placeholders below) is kept as a call, the adapter raises a `UserWarning` naming the setting
    to turn on, and the trace carries `meta["content_captured"] = False`. Order and name checks
    (`tool_called`, `tool_order`) still work.

    | Convention | Setting that captures content |
    |---|---|
    | OpenTelemetry GenAI instrumentations | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY` (or `SPAN_AND_EVENT`); the default is `NO_CONTENT`. Some instrumentations also need `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` |
    | Google ADK schema 2 | the same variable |
    | Google ADK schema 1 | `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS`; capture is on by default and an off setting writes `"{}"` for the arguments and the response |
    | OpenInference | unset `OPENINFERENCE_HIDE_INPUTS`, `OPENINFERENCE_HIDE_OUTPUTS`, `OPENINFERENCE_HIDE_INPUT_MESSAGES` and `OPENINFERENCE_HIDE_OUTPUT_MESSAGES`; hidden content is written as `__REDACTED__` |
    | Vercel AI SDK | the `recordInputs` and `recordOutputs` telemetry settings |

    Limits to know about: every call is turn 0, because no source says where a user turn begins;
    Google ADK schema 1 puts the reply somewhere this adapter does not read, so `output` is empty
    there; cached tokens are not mapped; and a trace with no tool spans cannot show that content
    was off, only that nothing was called.

## Next steps

* Write the checks that read the trace: [Testing agents](testing-agents.md#testing-agents).
* Run each case several times: [Repeated trials](repeated-trials.md#repeated-trials).
