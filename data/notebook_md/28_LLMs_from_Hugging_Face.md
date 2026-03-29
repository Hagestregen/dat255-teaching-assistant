<!-- source: 28_LLMs_from_Hugging_Face.ipynb -->

# Run open-source LLMs from the Hugging Face Hub

This mini-notebook just shows how you can download any model from Hugging Face, and run it as a chat assistant with practically no effort. You can find and search the full list of models at https://huggingface.co/models.

Remember that the memory requirements for the various models differ, so you may not be able to run all of them efficiently :)


```python
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
```


```python
pipe(messages)
```

