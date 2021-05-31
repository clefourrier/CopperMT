from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from data.management.from_etymdb.extract_data import get_inheritance_tree, get_children_relations
from data.management.from_etymdb.utils.language_info import ancestors