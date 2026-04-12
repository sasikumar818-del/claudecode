"""Tests for PluginRegistry — registration, lookup, and directory loading."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from plugins.base import AgentPlugin, BackendPlugin, BasePlugin
from plugins.registry import PluginRegistry


# ── Concrete plugin fixtures ──────────────────────────────────────────────────

class FakeBasePlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "fake_base"


class FakeAgentPlugin(AgentPlugin):
    @property
    def name(self) -> str:
        return "custom_script_writer"

    @property
    def replaces(self) -> str:
        return "Script Writer"

    def run(self, input_data):
        return MagicMock()


class FakeBackendPlugin(BackendPlugin):
    @property
    def name(self) -> str:
        return "fake_backend"

    @property
    def backend_type(self) -> str:
        return "image"

    def generate(self, prompt: str, **kwargs) -> bytes:
        return b"fake_image_bytes"


# ── Registration tests ────────────────────────────────────────────────────────

def test_register_and_lookup():
    registry = PluginRegistry()
    plugin = FakeBasePlugin()
    registry.register(plugin)
    assert registry.get_plugin("fake_base") is plugin


def test_register_duplicate_raises():
    registry = PluginRegistry()
    registry.register(FakeBasePlugin())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeBasePlugin())


def test_all_plugins_returns_all():
    registry = PluginRegistry()
    p1 = FakeBasePlugin()
    p2 = FakeBackendPlugin()
    registry.register(p1)
    registry.register(p2)
    plugins = registry.all_plugins()
    assert p1 in plugins
    assert p2 in plugins
    assert len(plugins) == 2


def test_get_plugin_returns_none_for_unknown():
    registry = PluginRegistry()
    assert registry.get_plugin("nonexistent") is None


# ── AgentPlugin lookup ────────────────────────────────────────────────────────

def test_get_agent_override_returns_agent_plugin():
    registry = PluginRegistry()
    plugin = FakeAgentPlugin()
    registry.register(plugin)
    result = registry.get_agent_override("Script Writer")
    assert result is plugin


def test_get_agent_override_returns_none_for_unregistered_step():
    registry = PluginRegistry()
    assert registry.get_agent_override("Script Writer") is None


def test_get_agent_override_returns_none_for_non_agent_plugin():
    registry = PluginRegistry()
    registry.register(FakeBasePlugin())
    # "fake_base" is not an AgentPlugin, so no override for any step name
    assert registry.get_agent_override("fake_base") is None


# ── BackendPlugin lookup ──────────────────────────────────────────────────────

def test_get_backend_returns_backend_plugin():
    registry = PluginRegistry()
    plugin = FakeBackendPlugin()
    registry.register(plugin)
    assert registry.get_backend("fake_backend") is plugin


def test_get_backend_returns_none_for_unknown():
    registry = PluginRegistry()
    assert registry.get_backend("nonexistent") is None


# ── Directory loading ─────────────────────────────────────────────────────────

def test_load_from_directory_with_nonexistent_dir_is_safe(tmp_path):
    registry = PluginRegistry()
    registry.load_from_directory(tmp_path / "does_not_exist")
    assert registry.all_plugins() == []


def test_load_from_directory_registers_concrete_plugin(tmp_path):
    plugin_code = textwrap.dedent("""\
        from plugins.base import BasePlugin

        class DynamicPlugin(BasePlugin):
            @property
            def name(self):
                return "dynamic_plugin"
    """)
    (tmp_path / "my_plugin.py").write_text(plugin_code)

    registry = PluginRegistry()
    registry.load_from_directory(tmp_path)
    assert registry.get_plugin("dynamic_plugin") is not None


def test_load_from_directory_skips_underscore_files(tmp_path):
    code = textwrap.dedent("""\
        from plugins.base import BasePlugin

        class PrivatePlugin(BasePlugin):
            @property
            def name(self):
                return "private_plugin"
    """)
    (tmp_path / "__init__.py").write_text(code)
    (tmp_path / "_helper.py").write_text(code)

    registry = PluginRegistry()
    registry.load_from_directory(tmp_path)
    assert registry.all_plugins() == []


def test_load_from_directory_skips_abstract_classes(tmp_path):
    """Concrete abstract subclasses (missing required methods) should not load."""
    code = textwrap.dedent("""\
        from plugins.base import AgentPlugin

        class IncompleteAgent(AgentPlugin):
            # Missing: name, replaces, run — still abstract
            pass
    """)
    (tmp_path / "incomplete.py").write_text(code)

    registry = PluginRegistry()
    registry.load_from_directory(tmp_path)
    assert registry.all_plugins() == []


def test_load_from_directory_handles_import_error_gracefully(tmp_path):
    bad_code = "import this_module_does_not_exist_at_all\n"
    (tmp_path / "broken.py").write_text(bad_code)

    registry = PluginRegistry()
    # Must not raise; should print a warning instead
    registry.load_from_directory(tmp_path)
    assert registry.all_plugins() == []


def test_load_from_directory_registers_agent_plugin(tmp_path):
    code = textwrap.dedent("""\
        from plugins.base import AgentPlugin

        class CustomWriter(AgentPlugin):
            @property
            def name(self):
                return "custom_writer"

            @property
            def replaces(self):
                return "Script Writer"

            def run(self, input_data):
                return None
    """)
    (tmp_path / "custom_writer.py").write_text(code)

    registry = PluginRegistry()
    registry.load_from_directory(tmp_path)
    override = registry.get_agent_override("Script Writer")
    assert override is not None
    assert override.name == "custom_writer"
