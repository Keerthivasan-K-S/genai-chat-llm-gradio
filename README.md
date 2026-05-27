# Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

## AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

## PROBLEM STATEMENT:

Building a user-friendly application that allows seamless interaction with a large language model (LLM) is challenging without requiring specialized API keys or external resources. This project addresses the need for an accessible, open-source solution to implement such applications using pre-trained models and the Gradio Blocks framework.


## DESIGN STEPS:

### STEP 1:

Import Required Libraries. Install and import the necessary libraries: Gradio for the UI. Transformers for using pre-trained models.

### STEP 2:

Use a pre-trained model like GPT-2 or DialoGPT from Hugging Face.Initialize the model using the pipeline API for straightforward interaction.

### STEP 3:

Run the application locally with demo.launch(). Optionally deploy it to the cloud for broader accessibility.

## PROGRAM:
```py
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading FLAN-T5 model...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
)

print("Model loaded successfully!")

def chat(message, history):

    inputs = tokenizer(
        message,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    reply = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": reply
        }
    )

    return "", history


with gr.Blocks() as demo:

    gr.Markdown("# 🤖 Chat with LLM")

    chatbot = gr.Chatbot()

    message = gr.Textbox(
        placeholder="Type your question..."
    )

    clear = gr.Button("Clear Chat")

    message.submit(
        chat,
        [message, chatbot],
        [message, chatbot]
    )

    clear.click(
        lambda: [],
        outputs=chatbot
    )

demo.launch()
```
## OUTPUT:
<img width="1918" height="928" alt="image" src="https://github.com/user-attachments/assets/0126aa95-c05b-4ee9-be61-1bba38f3572c" />


### RESULT:
Thus the Chat with LLM Application Using the Gradio Blocks Framework is created successfully.
