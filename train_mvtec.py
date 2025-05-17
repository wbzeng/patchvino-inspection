import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage
from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer, ExportType
from anomalib.engine import Engine
from anomalib.models import Patchvino
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode


# Create the datamodule in MVTec format
inputsize=300
root_dir = r"./xxx/mvtec_anomaly_detection/bottle" # Path to your dataset
datamodule = Folder(
    name="patchvino_rls", # Name of the datamodule
    root=root_dir,
    normal_dir=r"train/good",
    abnormal_dir=r"test",
    mask_dir=r"ground_truth",
    normal_test_dir=r"good",
    test_split_mode=TestSplitMode.FROM_DIR, 
    image_size=(inputsize, inputsize),
    normal_split_ratio = 0.0,
    val_split_ratio=0.1,
)
# Setup the datamodule
datamodule.setup()

# Setup the model
model=Patchvino(backbone="dinov2_vitb14",input_size=inputsize,coreset_sampling_ratio=0.1)
engine = Engine(accelerator="auto")
# Train the model
engine.fit(model=model,datamodule=datamodule)
# test the model
engine.test(datamodule=datamodule)


