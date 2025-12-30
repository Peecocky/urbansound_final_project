from models.crnn import build_crnn
from models.resnet18 import build_resnet18
from models.resnet34 import build_resnet34
from models.resnet50 import build_resnet50

MODEL_REGISTRY = {
    "crnn": build_crnn,
    "resnet18": build_resnet18,
    "resnet34": build_resnet34,
    "resnet50": build_resnet50,
}
