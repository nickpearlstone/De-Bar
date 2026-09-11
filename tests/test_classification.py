import importlib.util
import os
import sys

from dotenv import load_dotenv
from transformers import set_seed

spec = importlib.util.spec_from_file_location(
    "classify_image", "src/De-Bar/classification.py"
)
classification_module = importlib.util.module_from_spec(spec)
sys.modules["classify_image"] = classification_module
spec.loader.exec_module(classification_module)

# Extract your function
classify_image = classification_module.classify_image


# comment for testing purposes x2
def test_classify_image_letter():

    load_dotenv()

    set_seed(42)

    HF_TOKEN = os.getenv("HF_TOKEN")

    result = classify_image(token=HF_TOKEN, image_path="assets/images.jpg")

    print(f"Classification result: {result}")

    assert result == "letter"


def test_classify_image_form():

    load_dotenv()

    set_seed(42)

    HF_TOKEN = os.getenv("HF_TOKEN")

    result = classify_image(token=HF_TOKEN, image_path="assets/form-1040-1.png")

    print(f"Classification result: {result}")

    assert result == "form"
