from anomalib.data import MVTec
from anomalib.deploy import TorchInferencer
from anomalib.data.utils import read_image
from torch.utils.data import DataLoader
import torch
from anomalib.metrics import AUROC,AUPR,F1Score,F1Max
from tqdm import tqdm

# Create the datamodule
inputsize=300
root_dir = r"./xxx/mvtec_anomaly_detection" ## Path to your MVTec dataset
datamodule = MVTec(
    root=root_dir,
    category="bottle",
    image_size=inputsize,
    train_batch_size=0,  
    eval_batch_size=1,
    num_workers=1,
)
datamodule.setup()
test_dataloader = DataLoader(datamodule.test_data, batch_size=1, shuffle=False)
model_path = r"./xxx/model.pt" # Path to your trained model
torch.set_grad_enabled(mode=False)
inferencer = TorchInferencer(path=model_path, device="gpu")


# Initialize the metrics
pixel_auroc = AUROC()
image_auroc = AUPR()
pixel_f1_score = F1Score()
pixel_f1_max = F1Max()
image_f1_score = F1Score()
image_f1_max = F1Max()

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        inputs = batch["image"]
        masks = batch["mask"]
        labels = batch["label"]  

        filename = batch["image_path"]
        inputs = read_image(filename[0], as_tensor=True)
        predictions = inferencer.predict(inputs)

        # Pixel-level AUROC
        pixel_scores = predictions.anomaly_map
        pixel_auroc.update(torch.tensor(pixel_scores), torch.tensor(masks))

        # Image-level AUROC
        image_scores = predictions.pred_score
        image_auroc.update(torch.tensor(image_scores), torch.tensor(labels))

        # Pixel-level F1 Score
        pixel_f1_score.update(torch.tensor([pixel_scores]), torch.tensor(masks))
        pixel_f1_max.update(torch.tensor([pixel_scores]), torch.tensor(masks))

        # Image-level F1 Score
        image_f1_score.update(torch.tensor([image_scores]), torch.tensor(labels))
        image_f1_max.update(torch.tensor([image_scores]), torch.tensor(labels))

# Calculate the metrics
pixel_auroc_result = pixel_auroc.compute()
image_auroc_result = image_auroc.compute()
final_pixel_f1_score = pixel_f1_score.compute()
final_pixel_f1_max = pixel_f1_max.compute()
final_image_f1_score = image_f1_score.compute()
final_image_f1_max = image_f1_max.compute()

print(f"Image-level AUROC: {image_auroc_result.item()}")
print(f"Image-level F1 Score: {final_image_f1_score.item()}")
print(f"Pixel-level AUROC: {pixel_auroc_result.item()}")
print(f"Pixel-level F1 Score: {final_pixel_f1_score.item()}")
print(f"Pixel-level F1 Max: {final_pixel_f1_max.item()}")
print(f"Image-level F1 Max: {final_image_f1_max.item()}")

