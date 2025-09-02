import json
import os
import re
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


# ---- LangChain / GigaChat helpers ----
GIGA_CONFIG_PATH = "giga.yaml"


def _is_gigachat_model(model: str) -> bool:
    try:
        return "giga" in model.lower()
    except Exception:
        return False


def to_langchain_messages(messages: list[Message]):
    """
    Convert Tau2 messages to LangChain messages.
    Only user/assistant/system/tool content is mapped.
    """
    try:
        from langchain_core.messages import (
            HumanMessage as LCHumanMessage,
            AIMessage as LCAIMessage,
            SystemMessage as LCSystemMessage,
            ToolMessage as LCToolMessage,
            ToolCall as LCToolCall,
        )
    except Exception as e:
        raise ImportError(
            "LangChain is required for GigaChat. Install: pip install langchain langchain_gigachat"
        ) from e

    lc_messages: list[Any] = []
    for message in messages:
        if isinstance(message, UserMessage):
            lc_messages.append(LCHumanMessage(content=message.content))
        elif isinstance(message, AssistantMessage):
            # Preserve tool calls for LangChain so ToolMessages can be associated correctly
            if message.is_tool_call() and message.tool_calls is not None:
                lc_tool_calls = [
                    LCToolCall(
                        name=tc.name,
                        args=tc.arguments,
                        id=tc.id,
                        type="tool_call",
                    )
                    for tc in message.tool_calls
                ]
                lc_messages.append(LCAIMessage(content="", tool_calls=lc_tool_calls))
            else:
                lc_messages.append(LCAIMessage(content=message.content))
        elif isinstance(message, SystemMessage):
            lc_messages.append(LCSystemMessage(content=message.content))
        elif isinstance(message, ToolMessage):
            # Tool messages are passed through with their tool_call_id
            lc_messages.append(
                LCToolMessage(
                    content=message.content,
                    tool_call_id=getattr(message, "id", None),
                )
            )
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    return lc_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """

    # Transform Tool objects to dict objects
    if tools:
        tools = [
            tool.openai_schema for tool in tools
        ]

    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)

    if tools and tool_choice is None:
        tool_choice = "auto"

    # Route to LangChain GigaChat if requested
    if _is_gigachat_model(model):
        tools = [openai2giga_tool(tool) for tool in tools] if tools else None

        try:
            from yaml import safe_load
            from langchain_gigachat import GigaChat
        except Exception as e:
            raise ImportError(
                "Missing deps for GigaChat. Install: pip install langchain langchain_gigachat"
            ) from e

        # Load config from YAML
        try:
            with open(GIGA_CONFIG_PATH, "r") as f:
                config_giga: dict[str, str] = safe_load(f)["GigaChat"]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GigaChat config from {GIGA_CONFIG_PATH}: {e}"
            ) from e

        lc_messages = to_langchain_messages(messages)
        llm = GigaChat(**config_giga, timeout=200)
        if tools:
            llm = llm.bind_tools(tools=tools)
        # Retry on empty content/tool_calls up to N times
        empty_retry_attempts: int = int(kwargs.pop("empty_retry_attempts", 3))
        lc_response = None
        for empty_attempt in range(empty_retry_attempts):
            try:
                # Inner retry for transient invoke errors
                max_retries: int = 5
                for attempt in range(max_retries):
                    try:
                        lc_response = llm.invoke(lc_messages)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        logger.error(f"GigaChat invoke failed (attempt {attempt+1}/{max_retries}): {e}")
                        continue
            except Exception as e:
                logger.error(f"GigaChat invoke raised, empty-retry {empty_attempt+1}/{empty_retry_attempts}: {e}")
                continue

            # Map LangChain tool calls back to internal ToolCall objects if present
            lc_tool_calls: list[ToolCall] = []
            try:
                possible_calls = getattr(lc_response, "tool_calls", None)
                if possible_calls:
                    for call in possible_calls:
                        # Support both LC ToolCall objects and dict-like entries
                        name = getattr(call, "name", None) or call.get("name")
                        args = getattr(call, "args", None) or call.get("args", {})
                        call_id = getattr(call, "id", None) or call.get("id", "")
                        lc_tool_calls.append(
                            ToolCall(id=call_id or "", name=name, arguments=args)
                        )
            except Exception:
                lc_tool_calls = []

            content = getattr(lc_response, "content", None)
            has_content = content is not None and str(content).strip() != ""
            has_tool_calls = len(lc_tool_calls) > 0
        
            if has_content or has_tool_calls:
                message = AssistantMessage(
                    role="assistant",
                    content=content if has_content else None,
                    tool_calls=lc_tool_calls if has_tool_calls else None,
                    cost=0.0,
                    usage=None,
                    raw_data={"provider": "gigachat", "raw": str(lc_response)},
                )
                return message

            logger.warning(
                f"GigaChat returned empty response (attempt {empty_attempt+1}/{empty_retry_attempts}); retrying"
            )

        # If still empty after retries, return a placeholder to avoid validation errors
        logger.error("GigaChat returned empty response after retries; returning placeholder text")
        return AssistantMessage(
            role="assistant",
            content="Sorry, how can I help you, again?",
            tool_calls=None,
            cost=0.0,
            usage=None,
            raw_data={"provider": "gigachat", "raw": str(lc_response) if lc_response is not None else None},
        )

    # For other models (not GigaChat): use default LiteLLM implementation
    empty_retry_attempts: int = int(kwargs.pop("empty_retry_attempts", 3))
    last_exception: Exception | None = None
    last_raw: dict | None = None
    last_cost: float = 0.0
    last_usage: Optional[dict] = None

    if "gpt" in model.lower():
        # Set base_url for local OpenAI-compatible endpoint if not already set
        if "base_url" not in kwargs or kwargs["base_url"] is None:
            kwargs["base_url"] = os.environ.get("BASE_URL")

    for empty_attempt in range(empty_retry_attempts):
        try:
            response = completion(
                model=model,
                messages=litellm_messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
        except Exception as e:
            last_exception = e
            logger.warning(f"LiteLLM completion failed (empty-retry {empty_attempt+1}/{empty_retry_attempts}): {e}")
            continue
        last_cost = get_response_cost(response)
        last_usage = get_response_usage(response)
        choice = response.choices[0]
        try:
            finish_reason = choice.finish_reason
            if finish_reason == "length":
                logger.warning("Output might be incomplete due to token limit!")
        except Exception as e:
            logger.error(e)
            # don't fail here; proceed
        assert choice.message.role == "assistant", (
            "The response should be an assistant message"
        )
        content = choice.message.content
        tool_calls_raw = choice.message.tool_calls or []
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            )
            for tool_call in tool_calls_raw
        ]
        has_content = content is not None and str(content).strip() != ""
        has_tool_calls = len(tool_calls) > 0
        last_raw = response.to_dict()
        if has_content or has_tool_calls:
            return AssistantMessage(
                role="assistant",
                content=content if has_content else None,
                tool_calls=tool_calls if has_tool_calls else None,
                cost=last_cost,
                usage=last_usage,
                raw_data=last_raw,
            )
        logger.warning(
            f"LiteLLM returned empty response (attempt {empty_attempt+1}/{empty_retry_attempts}); retrying"
        )

    # After retries, fallback to placeholder to avoid validation error
    if last_exception is not None:
        logger.error(f"LiteLLM completion repeatedly failed; last error: {last_exception}")
    else:
        logger.error("LiteLLM returned empty response after retries; returning placeholder text")
    return AssistantMessage(
        role="assistant",
        content="(no content)",
        tool_calls=None,
        cost=last_cost,
        usage=last_usage,
        raw_data=last_raw,
    )


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage


def _replace_refs(obj, defs):
    if isinstance(obj, dict):
        # If this is a $ref, replace it
        if set(obj.keys()) == {"$ref"}:
            ref = obj["$ref"]
            if ref.startswith("#/$defs/"):
                name = ref.split("/")[-1]
                return defs[name]
        # Otherwise, recursively process all dict values
        return {k: _replace_refs(v, defs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_refs(item, defs) for item in obj]
    else:
        return obj

def _replace_anyof_with_object(schema):
    if isinstance(schema, dict):
        if "anyOf" in schema:
            # Replace "anyOf" with "type": "object"
            new_schema = schema.copy()

            # TODO: it's possible to ask an LMM to combine the schemas maybe...
            # For now: accept ANY schema if there are many valid
            possible_schemas = new_schema.pop("anyOf")

            if not possible_schemas:
                new_schema["type"] = "object"
                new_schema["properties"] = {}
                return new_schema

            for schema in possible_schemas:
                # The line below might seem weird. It is indeed.
                # One simple observation is that anyOf is always used with 1 $ref and 1 trivial empty object.
                # $ref represent a target object which is most likely to be found.
                # empty object is non-sensical and more of a `fallback` option.
                if len(str(schema)) > 100:
                    return schema

            new_schema["type"] = "object"
            new_schema["properties"] = {}
            return new_schema
        else:
            return {k: _replace_anyof_with_object(v) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [_replace_anyof_with_object(item) for item in schema]
    else:
        return schema

def openai2giga_tool(openai_tool: dict):
    func = openai_tool["function"]
    name = func["name"]
    description = func["description"]

    default_params = {
        "type": "object",
        "properties": {}
    }

    params = func.get("parameters", default_params)
    return_params = func.get("return_parameters", default_params)
    param_defs: dict[str, Any] = params.get("$defs", {})
    return_param_defs: dict[str, Any] = return_params.get("$defs", {})

    # Recursively replace $refs in parameters and return_parameters
    params_processed = _replace_refs(params, param_defs)
    params_processed = _replace_anyof_with_object(params_processed)
    # For return_parameters, if top-level is an object with a single $ref, resolve it
    if set(return_params.get("properties", {}).keys()) == {"returns"}:
        returns_schema = return_params["properties"]["returns"]
        if "$ref" in returns_schema:
            ref = returns_schema["$ref"]
            if ref.startswith("#/$defs/"):
                name_ = ref.split("/")[-1]
                resolved = return_param_defs[name_]
                # Recursively resolve any $refs inside the resolved schema
                resolved = _replace_refs(resolved, return_param_defs)
                resolved = _replace_anyof_with_object(resolved)
                return_params_processed = {
                    "properties": {"returns": resolved},
                    "required": return_params.get("required", []),
                }
            else:
                return_params_processed = {
                    "properties": return_params.get("properties", {}),
                    "required": return_params.get("required", []),
                }
        else:
            return_params_processed = {
                "properties": return_params.get("properties", {}),
                "required": return_params.get("required", []),
            }
    else:
        return_params_processed = {
            "properties": return_params.get("properties", {}),
            "required": return_params.get("required", []),
        }
    return_params_processed = _replace_anyof_with_object(return_params_processed)

    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": params_processed["properties"],
            "required": params_processed.get("required", [])
        },
        "return_parameters": {
            "type": "object",
            "properties": return_params_processed["properties"],
            "required": return_params_processed.get("required", [])
        }
    }
