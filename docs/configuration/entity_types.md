# Configuring Entity Types

There are three main approaches that we've considered to configure entity types in PoorMan's GraphRAG. Each approach offers different trade-offs between simplicity, flexibility, and runtime configurability. We document the approaches here for a future implementation.

## 1. Configuration File Approach

The simplest approach using YAML configuration files.

### Implementation

```python
from pathlib import Path
import yaml
from typing import Literal

def load_entity_types(config_path: Path) -> Literal:
    """Load entity types from a YAML configuration file."""
    with config_path.open() as f:
        config = yaml.safe_load(f)
    return Literal[tuple(config["entity_types"])]

EntityType = load_entity_types(Path("config/entity_types.yaml"))
```

### Configuration File Example (`config/entity_types.yaml`)

```yaml
entity_types:
  - paper
  - author
  - gene
  - protein
  # ... other types ...
```

**Pros:**
- Simple to implement and understand
- Easy for users to modify without code changes
- Configuration can be version controlled

**Cons:**
- Requires application restart to change types
- Less type safety at runtime
- No programmatic type registration

## 2. Registry Pattern Approach

A more flexible approach that allows runtime registration of entity types.

### Implementation

```python
class EntityTypeRegistry:
    """Registry for managing available entity types."""

    def __init__(self) -> None:
        self._types: Set[str] = set()

    def register(self, entity_type: str) -> None:
        """Register a new entity type."""
        self._types.add(entity_type)

    def get_types(self) -> Literal:
        """Get registered types as a Literal type."""
        return Literal[tuple(sorted(self._types))]

# Global registry instance
entity_registry = EntityTypeRegistry()
EntityType = entity_registry.get_types()
```

**Pros:**
- Types can be added/removed at runtime
- Programmatic control over type registration
- Good for dynamic type systems

**Cons:**
- More complex implementation
- Requires state management
- Global state can be tricky to manage

## 3. Plugin System Approach

The most flexible approach, allowing third-party extensions.

### Implementation

```python
@runtime_checkable
class EntityTypePlugin(Protocol):
    """Protocol for entity type plugins."""

    @property
    def entity_types(self) -> List[str]:
        """Get entity types provided by this plugin."""
        ...

def load_plugins(plugin_dir: Path) -> Literal:
    """Load entity types from plugins."""
    types = set()
    for plugin_file in plugin_dir.glob("*.py"):
        if plugin_file.name.startswith("_"):
            continue
        module = import_module(f"poorman_graphrag.plugins.{plugin_file.stem}")
        if hasattr(module, "plugin") and isinstance(module.plugin, EntityTypePlugin):
            types.update(module.plugin.entity_types)
    return Literal[tuple(sorted(types))]

EntityType = load_plugins(Path(__file__).parent / "plugins")
```

### Example Plugin (`plugins/biomedical.py`)

```python
class BiomedicalTypes:
    @property
    def entity_types(self) -> List[str]:
        return ["gene", "protein", "disease", "drug"]

plugin = BiomedicalTypes()
```

**Pros:**
- Most flexible and extensible
- Allows third-party contributions
- Clean separation of concerns

**Cons:**
- Most complex to implement
- Requires careful plugin management
- Need to handle plugin conflicts

## Usage with Entity Class

All approaches integrate with the `Entity` class through the `valid_types()` class method:

```python
class Entity(BaseModel):
    @classmethod
    def valid_types(cls) -> tuple[str, ...]:
        """Get currently valid entity types."""
        return get_args(EntityType)

    @model_validator(mode="before")
    def validate_entity_type(cls, data: Any) -> Any:
        """Validate that entity_type is one of the configured types."""
        if isinstance(data, dict) and "entity_type" in data:
            if data["entity_type"] not in cls.valid_types():
                raise ValueError(
                    f"Invalid entity_type: {data['entity_type']}. "
                    f"Must be one of: {cls.valid_types()}"
                )
        return data
```

## Recommendation

Start with the Configuration File approach unless you have specific needs for runtime flexibility or plugin support. It provides a good balance of simplicity and usability while being sufficient for most use cases.

If you need runtime type registration, consider the Registry Pattern. Only move to the Plugin System if you need to support third-party extensions or have complex type management requirements.
