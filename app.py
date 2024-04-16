import gradio as gr
from utils import T5Generator


# Preprocessing function that modifies the user input.
def preprocess_input(user_input):
    bos_instruction_id = """Definition: The output will be the aspects (both implicit and explicit) and the aspect's sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
            Positive example 1-
            input: With the great variety on the menu , I eat here often and never get bored.
            output: menu:positive
            Positive example 2-
            input: Great food, good size menu, great service and an unpretentious setting.
            output: food:positive, menu:positive, service:positive, setting:positive
            Negative example 1-
            input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.
            output: toast:negative, mayonnaise:negative, bacon:negative, ingredients:negative, plate:negative
            Negative example 2-
            input: The seats are uncomfortable if you are sitting against the wall on wooden benches.
            output: seats:negative
            Neutral example 1-
            input: I asked for seltzer with lime, no ice.
            output: seltzer with lime:neutral
            Neutral example 2-
            input: They wouldn't even let me finish my glass of wine before offering another.
            output: glass of wine:neutral
            Now complete the following example-
            input: """
    eos_instruction = ' \noutput:'
    # Append and prepend the text to the user's input.
    modified_input = bos_instruction_id + user_input + eos_instruction
    return modified_input


# Assuming you have a function `model_predict` that takes the processed text and returns a prediction.
def model_predict(processed_text):
    # Here you would include the logic to make a prediction based on the processed_text.
    # This example simply returns the processed text.
    # tokenize input
    input_ids = t5_exp.tokenizer(processed_text, return_tensors="pt").input_ids
    # generate output
    outputs = t5_exp.model.generate(input_ids, max_length=128)
    return t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Create T5 model object along with instructions
model_checkpoint = "Homeskills/mt_instruct_absa"
t5_exp = T5Generator(model_checkpoint)

iface = gr.Interface(fn=lambda x: model_predict(preprocess_input(x)),
                     inputs=[gr.Textbox(label="Input review text here")],
                     outputs=gr.Textbox(label="Extracted aspects & sentiment"),
                     title="Aspect-Based Sentiment Analysis (ABSA)",
                     description="This app will take a sample review segment as input and return the extracted aspects and the corresponding sentiments.",
                     article="This model is a fine-tuned version of the original SOTA model found here: https://huggingface.co/kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined",
                     )
iface.launch()
