"""
SAHA – Tool Runner: executes tool calls in a sandboxed environment.
In Phase 1 we support a registry of simple Python-callable tools.
Spec ref: §3.3 (NEEDS_TOOL handling)
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from saha.contracts.common import CanonicalError, ErrorCode, ErrorSeverity, ErrorType

logger = logging.getLogger(__name__)

# Type alias for tool functions
ToolFn = Callable[..., Coroutine[Any, Any, Any]]

# Default timeout per tool call (seconds)
TOOL_TIMEOUT_SECONDS = 30


class ToolResult:
    def __init__(
        self,
        tool_name: str,
        success: bool,
        output: Any,
        error: CanonicalError | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.success   = success
        self.output    = output
        self.error     = error or CanonicalError.none()

    def as_message_content(self) -> str:
        """Format result as a string to inject back into the conversation."""
        if self.success:
            return f"[Tool:{self.tool_name}] Result: {self.output}"
        return f"[Tool:{self.tool_name}] Error: {self.error.details}"


class ToolRunner:
    """
    Maintains a registry of available tools and executes them asynchronously.
    Each tool is an async callable: async def tool_fn(**kwargs) -> Any
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn
        logger.info("Registered tool: %s", name)

    async def run(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=CanonicalError(
                    type=ErrorType.TOOL_ERROR,
                    code=ErrorCode.INVALID_TOOL_INPUT,
                    severity=ErrorSeverity.WARNING,
                    details=f"Unknown tool: '{tool_name}'",
                ),
            )

        fn = self._tools[tool_name]
        try:
            output = await asyncio.wait_for(fn(**arguments), timeout=TOOL_TIMEOUT_SECONDS)
            return ToolResult(tool_name=tool_name, success=True, output=output)
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=CanonicalError(
                    type=ErrorType.TOOL_ERROR,
                    code=ErrorCode.TOOL_TIMEOUT,
                    severity=ErrorSeverity.WARNING,
                    details=f"Tool '{tool_name}' timed out after {TOOL_TIMEOUT_SECONDS}s",
                ),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=CanonicalError(
                    type=ErrorType.TOOL_ERROR,
                    code=ErrorCode.TOOL_EXECUTION_FAILED,
                    severity=ErrorSeverity.WARNING,
                    details=f"Tool '{tool_name}' raised: {exc}",
                ),
            )


# ─── Built-in demo tools (Phase 1) ──────────────────────────────────────────

async def _tool_echo(message: str = "") -> str:
    """Simple echo – useful for testing the harness."""
    return f"ECHO: {message}"


async def _tool_get_time() -> str:
    """Return current UTC time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def build_default_tool_runner() -> ToolRunner:
    runner = ToolRunner()
    runner.register("echo", _tool_echo)
    runner.register("get_time", _tool_get_time)
    return runner
