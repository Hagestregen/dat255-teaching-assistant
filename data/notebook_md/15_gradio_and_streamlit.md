<!-- source: 15_gradio_and_streamlit.ipynb -->

# Gradio (and Streamlit) deployment

For deploying an ML-based app there are various approaches, but [Gradio](gradio.app) and [Streamlit](https://streamlit.io/) are both quick and convenient ways to do so. Typically we would implement this in a plain python (`.py`) file rather than a `.ipynb` notebook, but Gradio works here too, so let's try that first. Streamlit needs to be run directly in python, but the code is given below, so you can try out that too.

In this example we set up an image classifier, where the user can upload an image and get the top 5 class predictions in return.


```python
import numpy as np
import tensorflow as tf
from tf.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image
```


## Set up a pretrained model

Let's download and set up a `MobileNetV2` model, trained on the 1000 classes in the ImageNet dataset. You can change this to anything you like. Remeber, however, to also use the appropriate preprocessing function.


```python
model = MobileNetV2(weights="imagenet")

def classify_image(img: Image.Image):

    # Resize to the input image to what MobileNet expects
    img = img.resize((224, 224))
    arr = np.array(img)

    # Check the color channels, so we can take both grayscale, RGB, and RGBA images as input.
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    # Add batch dim, and preprocess
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr.astype("float32"))

    # Predict!
    preds = model.predict(arr, verbose=0)
    top5 = decode_predictions(preds, top=5)[0]  # [(id, label, prob), ...]

    return {label: float(prob) for (_, label, prob) in top5}
```


## Set up the Gradio interface

Check the [documentation](https://www.gradio.app/docs) for the various things we can add here.


```python
import gradio as gr

# Example images
examples = [
    "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg",
    "https://cdn.britannica.com/41/126641-050-E1CA0E61/cat-suns-hill-Parthenon-Athens-Greece-Acropolis.jpg",
]

with gr.Blocks(title="Keras Image Classifier") as demo:
    gr.Markdown("## Image classifier")
    gr.Markdown(
        "Upload an image or click an example below. "
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image")
        label_output = gr.Label(num_top_classes=5, label="Predictions")

    classify_btn = gr.Button("Classify", variant="primary")
    classify_btn.click(fn=classify_image, inputs=image_input, outputs=label_output)

    gr.Examples(
        examples=examples,
        inputs=image_input,
        outputs=label_output,
        fn=classify_image,
        cache_examples=True
    )
```


Now we can run it:


```python
demo.launch()
```


## Streamlit alternative

Because of the way Streamlit runs its own web server, it has to be run outside of a notebook. To try it out, copy the code below into a python file (for instance `app.py`), and run it like so:

```
pip install streamlit

streamlit run app.py
```


```python
import streamlit as st

st.set_page_config(page_title="Keras Image Classifier", page_icon="🖼️")
st.title("Image classifier")
st.write("Upload an image or click an example.")


@st.cache_resource
def load_model():
    """ Cache the model to avoid reloading """
    return MobileNetV2(weights="imagenet")

model = load_model()


def classify(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)

    preds = model.predict(preprocess(img), verbose=0)
    top5 = decode_predictions(preds, top=5)[0]

    return [(label, float(prob)) for (_, label, prob) in top5]


def load_url_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert("RGB")

# Examples
examples = {
    "Dog": "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg",
    "Cat": "https://cdn.britannica.com/41/126641-050-E1CA0E61/cat-suns-hill-Parthenon-Athens-Greece-Acropolis.jpg",
}

# Make a sidebar with example images
st.sidebar.header("Example images")
example_choice = st.sidebar.radio("Click an example to load it:", ["— none —"] + list(examples.keys()))


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

# Decide which image to use: upload takes priority over example
img: Image.Image | None = None

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
elif example_choice != "— none —":
    with st.spinner("Loading example image…"):
        img = load_url_image(examples[example_choice])

# Display image and predictions.
if img:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Predictions")
        with st.spinner("Predicting..."):
            results = classify(img)

        for label, prob in results:
            st.write(f"**{label.replace('_', ' ').title()}**")
            st.progress(prob, text=f"{prob:.1%}")
else:
    st.info("Upload an image or click an example.")
```

