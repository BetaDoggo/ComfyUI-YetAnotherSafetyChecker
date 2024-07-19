from transformers import pipeline
from torchvision import transforms
import torch

class YetAnotherSafetyChecker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "cuda": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    FUNCTION = "process_images"

    CATEGORY = "image/processing"

    def process_images(self, image, threshold, cuda):
        if cuda:
            device = "cuda"
        else:
            device = "cpu"
        predict = pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector", device=device) #init pipeline
        result = (predict(transforms.ToPILImage()(image[0].cpu().permute(2, 0, 1)))) #Convert to expected format
        score = next(item['score'] for item in result if item['label'] == 'nsfw')
        output = image
        if(float(score) > threshold):
            output = torch.zeros(1, 512, 512, dtype=torch.float32) #create black image tensor
        return (output, image, str(score))
    
NODE_CLASS_MAPPINGS = {
    "YetAnotherSafetyChecker": YetAnotherSafetyChecker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YetAnotherSafetyChecker": "Intercept NSFW Outputs"
}
