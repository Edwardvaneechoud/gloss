"""Apache Ossie interop: import documents as proposals, export accepted facts as documents.

``ossie-schema.json`` is the Apache Ossie core JSON Schema (Apache License 2.0),
vendored unchanged for validation. See NOTICE.
"""

GLOSS_VENDOR = "gloss"
OSSIE_VERSION = "0.3.0.dev0"

from .exporter import OssieSink, to_ossie, validate  # noqa: E402
from .importer import OssieSource, load_document  # noqa: E402

__all__ = ["GLOSS_VENDOR", "OSSIE_VERSION", "OssieSink", "OssieSource", "load_document", "to_ossie", "validate"]
