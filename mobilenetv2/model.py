import torch
import requests
import numpy as np
import onnxruntime as ort
from typing import List
from torchvision import transforms
from PIL import Image
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_classification_to_vision_input,
    construct_task_classification_output,
)


@instill_deployment
class MobileNet:
    def __init__(self):
        self.categories = self._image_labels()
        self.model = ort.InferenceSession("model.onnx")
        self.tf = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _image_labels(self) -> List[str]:
        categories = []
        url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        labels = requests.get(url, timeout=10).text
        for label in labels.split("\n"):
            categories.append(label.strip())
        return categories

    async def __call__(self, request):
        vision_inputs = parse_task_classification_to_vision_input(request=request)

        batch_out = []
        for inp in vision_inputs:
            image = np.array(inp.image)
            np_tensor = self.tf(Image.fromarray(image, mode="RGB")).numpy()
            batch_out.append(np_tensor)

        batch_out = np.asarray(batch_out)
        out = self.model.run(None, {"input": batch_out})
        # shape=(1, batch_size, 1000)

        # tensor([[207], [294]]), tensor([[0.7107], [0.7309]])
        score, cat = torch.topk(torch.from_numpy(out[0]), 1)

        scores = [score[i][0] for i in range(cat.size(0))]
        categories = [self.categories[cat[i]] for i in range(cat.size(0))]

        return construct_task_classification_output(
            categories=categories, scores=scores
        )


entrypoint = InstillDeployable(MobileNet).get_deployment_handle()
