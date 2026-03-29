<!-- source: 25_attention_visualisation.ipynb -->

# Visualise the attention scores in an LLM

Let us visualise the attention scores in different language models, using a tool called [BertViz](https://github.com/jessevig/bertviz). For a thorough explanation of the visualisations, have a look at the accompanying [paper](https://aclanthology.org/P19-3007.pdf).

This notebook is basically a ripoff from the BertViz tutorial, which is a great resource, but here we add some newer models.


First, install the prerequisites.


```python
!pip install --upgrade transformers
!pip install --upgrade huggingface_hub
!pip install bertviz
```


## Select a model

Using the Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) library, we can run models publicly available on the Hugging Face [Hub](https://huggingface.co/models). For some models (like Google's Gemma and Meta's Llama), you need a make a user account and log in before being able to download them, but most can be downloaded directly.

For our first test we choose the good old GTP-2 model. We can search it up and find it on the Hugging Face Hub here: https://huggingface.co/openai-community/gpt2.

Then copy the name/model tag and automatically download and set up both the model, and the accompanying tokeniser.


```python
from transformers import AutoTokenizer, AutoModel, utils

model_name = 'openai-community/gpt2'
#model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
#model_name = 'microsoft/Phi-4-mini-instruct'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
```


### Select theme

Maybe the most important step???

Select dark or light theme for the plots.


```python
theme = 'dark'
#theme = 'light'
```


## Plot attention scores

To investigate the attention scores, we need some text to process. You can try out any sentence you like.

Then run and plot the attention weight for each attention weights in each layer.


```python
input_sentence = "Deep learning is cool! Do you think so too?"

inputs = tokenizer.encode(input_sentence, return_tensors='pt')
outputs = model(inputs)

attention = outputs[-1]
tokens = tokenizer.convert_ids_to_tokens(inputs[0])

from bertviz import head_view
head_view(attention, tokens)
```


### Interpret results

By default, the attention scores for all heads are shown at the same time, which might be a bit of information overload. Select only one by double-clicking one of the colored boxes.

Now you can hover over the words, to show the attention scores between them.

### <span style="color: red;">Exercise:<span>

Try to investigate the model: Are the different attention heads and the different layers focusing on different information?


### Plot all at once

To ease the comparison, we can plot all layers and attention heads in a matrix.
Click any of them to enlarge the plot.


```python
from bertviz import model_view
model_view(attention, tokens, display_mode=theme)
```


## Neuron view

For the final piece of magic -- let's visualise how the attention scores are computed.

Hover over one of the words and click the plus (+) to open a plot of the query and key vectors, and the product of the two.

**Note:** This only works for selected models that are implemented in BertViz.


```python
from bertviz.transformers_neuron_view import GPT2Model, GPT2Tokenizer
from bertviz.neuron_view import show

model_type = 'gpt2'
model_version = 'gpt2'
bertviz_model = GPT2Model.from_pretrained(model_version, output_attentions=True)
bertviz_tokenizer = GPT2Tokenizer.from_pretrained(model_version, do_lower_case=True)
show(bertviz_model, model_type, bertviz_tokenizer, input_sentence, layer=4, head=3, display_mode=theme)
```


### <span style="color: red;">Exercise:<span>

Find and investigate other models -- maybe one of the recent DeepSeek versions?

