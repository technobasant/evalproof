# Agent SDK adapters

A python system (`target: python:agent:run`) returns `{"output", "trace", ...}`, and the `trace`
is what the `checks` scorer reads: the tools called, their arguments and results, the rounds,
the tokens. `evalproof.adapters` builds that dict from what four agent SDKs return. The
adapters import no SDK and read objects or plain dicts, so they work with any version that keeps
the field names they cite; each module's header gives the package version and file:line of every
field it reads. Tokens and cost roll up from the trace, so `max_cost_usd` and the budgets see an
agent's spend. Cost is set only where the SDK reports one (Claude); otherwise the target's
`pricing` applies. A turn boundary is read from the input where it shows (ADK user events, a
LangChain human message, a Claude `ResultMessage`), else every call is turn 0. For a multi-turn
case where each user turn is a separate call to the SDK, pass one item per turn: a list of ADK
event lists, a list of Claude message lists (one per `query()` call), or a list of OpenAI
results. The python function gets the later turns as `ctx.turns`.

**Google ADK** — `adk.from_events(events)`; pairs `function_call` and `function_response` parts
by id, then by name and order. A tool that returns `{"error": ...}` is recorded as failed.
`run_async` does not yield the user's message by default, so for several turns pass one event
list per run (or `session.events`).

```python
# agent.py
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

**LangChain and LangGraph** — `langchain.from_messages(messages)` takes the message list or the
state dict an agent returns; `ToolMessage.status == "error"` marks a failed call.

```python
# agent.py
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

**Claude Agent SDK** — `claude_agent_sdk.from_messages(messages)`; `ToolResultBlock.is_error`
marks a failed call, the usage comes from the per-model `model_usage` of the final
`ResultMessage`, and `cost_usd` is its `total_cost_usd`. Those are running totals of a session;
pass one message list per `query()` call and the sessions' totals are added.

```python
# agent.py
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

**OpenAI Agents SDK** — `openai_agents.from_result(result)` takes a `RunResult`, or a list of
them for a multi-turn case (one result per user turn). The SDK has no field that says a tool
failed: it turns an exception into a plain string, by default the fixed text in
`openai_agents.DEFAULT_TOOL_ERROR`, and an output equal to that text is recorded as failed. For a
tool with its own `failure_error_function`, pass `error_if=lambda output: ...`, which replaces
the default test (`error_if=lambda output: False` turns it off).

```python
# agent.py
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

**Plain Python, or MCP** — `TraceRecorder` builds the same dict by hand, and `mcp_call` records
an MCP `CallToolResult` (`isError` makes the call fail):

```python
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

**TypeScript (Vercel AI SDK).** An app in another language joins through an `http` target
(`target: http://localhost:3000/eval`). evalproof POSTs `{"id", "input"}` and reads
`{"output", "trace", ...}` back, with the `trace` in evalproof's field names, so the app maps
`generateText`'s `text`, `toolCalls`, `toolResults` and `usage` (all steps, in `ai` 7) itself:

```ts
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
