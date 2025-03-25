
import json
import torch
from torchvision.transforms import v2 as vision_transforms_v2
from PIL import Image
import io
import gradio as gr


def load_transforms():
    return vision_transforms_v2.Compose([
        vision_transforms_v2.ToImage(),

        vision_transforms_v2.ToDtype(torch.uint8, scale=True),
        vision_transforms_v2.Resize((224, 224), antialias=True),

        vision_transforms_v2.ToDtype(torch.float32, scale=True),
        vision_transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_class_names(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def load_model(filename):
    model = torch.load(filename, weights_only=False, map_location='cpu')
    model.eval()
    return model


def get_prediction(image, transforms, model, label_to_class):
    image_input = transforms(image).unsqueeze(0)
    outputs = model(image_input)
    predicted = outputs.argmax(dim=1).cpu().numpy()
    label = predicted.item()
    return label, label_to_class[label]


def main():
    label_to_class = load_class_names('label_to_class.json')
    model = load_model('production-model.pt')
    transforms = load_transforms()

    fn = lambda image: get_prediction(image, transforms, model, label_to_class)

    interface = gr.Interface(
        fn=fn,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Number(label="Class ID"),
            gr.Textbox(label="Class Name")
        ],
        title="Image Classification Demo",
        description="Upload an image to classify it using a trained model",
    )
    interface.launch(share=True)

if __name__ == '__main__':
    main()
