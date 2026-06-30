"""
SAHA – Unit tests for ToolRunner (§3.3).
"""
import asyncio
import pytest

from saha.contracts.common import ErrorCode, ErrorType
from saha.execution.tool_runner import ToolRunner, build_default_tool_runner


@pytest.fixture
def runner() -> ToolRunner:
    return build_default_tool_runner()


class TestToolRunnerBuiltins:
    async def test_echo_tool(self, runner: ToolRunner) -> None:
        result = await runner.run("echo", {"message": "hello"})
        assert result.success is True
        assert "ECHO: hello" in result.output

    async def test_get_time_tool(self, runner: ToolRunner) -> None:
        result = await runner.run("get_time", {})
        assert result.success is True
        assert "T" in result.output  # ISO 8601 format

    async def test_unknown_tool(self, runner: ToolRunner) -> None:
        result = await runner.run("nonexistent_tool", {})
        assert result.success is False
        assert result.error.type == ErrorType.TOOL_ERROR
        assert result.error.code == ErrorCode.INVALID_TOOL_INPUT


class TestToolRunnerCustom:
    async def test_custom_tool_registered(self) -> None:
        runner = ToolRunner()

        async def add(a: int, b: int) -> int:
            return a + b

        runner.register("add", add)
        result = await runner.run("add", {"a": 3, "b": 7})
        assert result.success is True
        assert result.output == 10

    async def test_tool_exception_caught(self) -> None:
        runner = ToolRunner()

        async def boom(**kwargs) -> None:
            raise ValueError("test error")

        runner.register("boom", boom)
        result = await runner.run("boom", {})
        assert result.success is False
        assert result.error.type == ErrorType.TOOL_ERROR
        assert result.error.code == ErrorCode.TOOL_EXECUTION_FAILED
        assert "test error" in result.error.details

    async def test_timeout_enforced(self) -> None:
        runner = ToolRunner()
        # Monkey-patch timeout to 0.01s for speed
        import saha.execution.tool_runner as tr_mod
        original = tr_mod.TOOL_TIMEOUT_SECONDS
        tr_mod.TOOL_TIMEOUT_SECONDS = 0.01

        async def slow(**kwargs) -> None:
            await asyncio.sleep(10)

        runner.register("slow", slow)
        result = await runner.run("slow", {})
        assert result.success is False
        assert result.error.code == ErrorCode.TOOL_TIMEOUT

        tr_mod.TOOL_TIMEOUT_SECONDS = original

    def test_as_message_content_success(self) -> None:
        from saha.execution.tool_runner import ToolResult
        r = ToolResult("my_tool", True, "result_data")
        msg = r.as_message_content()
        assert "my_tool" in msg
        assert "result_data" in msg

    def test_as_message_content_failure(self) -> None:
        from saha.execution.tool_runner import ToolResult
        from saha.contracts.common import CanonicalError
        err = CanonicalError.critical(ErrorType.TOOL_ERROR, ErrorCode.TOOL_TIMEOUT, "timed out")
        r = ToolResult("slow_tool", False, None, error=err)
        msg = r.as_message_content()
        assert "Error" in msg
        assert "timed out" in msg
