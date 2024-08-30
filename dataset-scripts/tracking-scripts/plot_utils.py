from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import numpy as np

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]
    
    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
    
    @property
    def width(self) -> int:
        return self.xmax - self.xmin
    
    @property
    def height(self) -> int:
        return self.ymax - self.ymin

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))