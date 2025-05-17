from anomalib import TaskType
from anomalib.deploy import ExportType,CompressionType
from anomalib.engine import Engine
from anomalib.models import Patchvino
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize

# Exporting model to OpenVINO
engine = Engine(accelerator="auto")
inputsize = 300
model=Patchvino(backbone="dinov2_vitb14",input_size=inputsize,coreset_sampling_ratio=0.1)

openvino_model_path = engine.export(
    model=model,
    export_type=ExportType.TORCH,
    ckpt_path='./xxx/xxx/model.ckpt', # Path to your checkpoint
    export_root='./exported_model_PatchVino', # Path to save the exported model
)