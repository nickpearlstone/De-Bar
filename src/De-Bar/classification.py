from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def classify_image(token: str, image_path: str = "De-Bar/assets/images.jpg"):

    image = Image.open(image_path).convert("RGB")

    processor = AutoImageProcessor.from_pretrained(
        "microsoft/dit-base-finetuned-rvlcdip", token=token
    )
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/dit-base-finetuned-rvlcdip", token=token
    )

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # model predicts one of the 16 RVL-CDIP classes
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]
