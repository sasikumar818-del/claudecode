"""Plugin registry — registers and discovers pipeline plugins.

Usage
-----
Instantiate a :class:`PluginRegistry`, optionally call
:meth:`load_from_directory` to auto-discover plugins from a folder, then pass
the registry to :class:`~agents.supervisor.SupervisorAgent`.

    registry = PluginRegistry()
    registry.load_from_directory(Path("plugins"))
    supervisor = SupervisorAgent(settings, registry=registry)

You can also register plugins manually:

    registry.register(MyCustomPlugin())
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

from plugins.base import AgentPlugin, BackendPlugin, BasePlugin


class PluginRegistry:
    """Central store for pipeline plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, BasePlugin] = {}
        # AgentPlugins indexed by the step name they replace
        self._agent_overrides: dict[str, AgentPlugin] = {}
        # BackendPlugins indexed by plugin name
        self._backends: dict[str, BackendPlugin] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin: BasePlugin) -> None:
        """Register a plugin instance.

        Raises :exc:`ValueError` if a plugin with the same name is already
        registered.
        """
        if plugin.name in self._plugins:
            raise ValueError(
                f"A plugin named '{plugin.name}' is already registered."
            )
        self._plugins[plugin.name] = plugin
        if isinstance(plugin, AgentPlugin):
            self._agent_overrides[plugin.replaces] = plugin
        if isinstance(plugin, BackendPlugin):
            self._backends[plugin.name] = plugin

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Return the plugin with *name*, or ``None``."""
        return self._plugins.get(name)

    def get_agent_override(self, step_name: str) -> Optional[AgentPlugin]:
        """Return an :class:`AgentPlugin` that replaces *step_name*, or ``None``."""
        return self._agent_overrides.get(step_name)

    def get_backend(self, name: str) -> Optional[BackendPlugin]:
        """Return a :class:`BackendPlugin` by *name*, or ``None``."""
        return self._backends.get(name)

    def all_plugins(self) -> list[BasePlugin]:
        """Return all registered plugins in registration order."""
        return list(self._plugins.values())

    # ------------------------------------------------------------------
    # Directory-based discovery
    # ------------------------------------------------------------------

    def load_from_directory(self, plugin_dir: Path) -> None:
        """Import every ``*.py`` file in *plugin_dir* and register any
        :class:`BasePlugin` subclasses found inside.

        Files whose names begin with ``_`` (including ``__init__.py``) are
        skipped.  Import errors are printed but do not abort loading.
        """
        if not plugin_dir.exists():
            return
        for py_file in sorted(plugin_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            self._load_plugin_file(py_file)

    def _load_plugin_file(self, path: Path) -> None:
        module_name = f"_user_plugins.{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                print(f"  [PluginRegistry] Could not load spec from {path}")
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            print(f"  [PluginRegistry] Error importing {path.name}: {exc}")
            return

        # Find concrete BasePlugin subclasses defined in this module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BasePlugin)
                and attr not in (BasePlugin, AgentPlugin, BackendPlugin)
                and not getattr(attr, "__abstractmethods__", None)
            ):
                try:
                    instance = attr()
                    self.register(instance)
                    print(
                        f"  [PluginRegistry] Loaded plugin '{instance.name}' "
                        f"from {path.name}"
                    )
                except Exception as exc:
                    print(
                        f"  [PluginRegistry] Failed to instantiate "
                        f"{attr_name} from {path.name}: {exc}"
                    )
