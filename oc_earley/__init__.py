import sys

# kernels is not reexported as it should remain an optional dependency
from . import _json_schema as json_schema
from .oc_earley import CompiledSchema, Guide, Index, Recognizer, Vocabulary

# Register json_schema in sys.modules so "from oc_earley.json_schema
# import ..." works
sys.modules["oc_earley.json_schema"] = json_schema
