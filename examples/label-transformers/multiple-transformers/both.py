import sys
from pathlib import Path

from bbox_transformer import BBoxTransformer
from polygon_transformer import PolyTransformer

p = Path(__file__).absolute().parent
sys.path.append(p.as_posix())
