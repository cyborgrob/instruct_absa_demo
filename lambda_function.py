from InstructABSA.utils import T5Generator

print('Mode set to: Individual sample inference')


# Create T5 model object along with instructions (taken from `instructions.py`)
model_checkpoint = "./Models/joint_task/kevinscariajoint_tk-instruct-base-def-pos-neg-neut-combined-robs_experiment"
t5_exp = T5Generator(model_checkpoint)
print("Model loaded from: ", model_checkpoint)
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

# Get input from user
user_input = input("Enter sentence for inference: ")
# format and tokenize input
model_input = bos_instruction_id + user_input + eos_instruction
input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
# generate output
outputs = t5_exp.model.generate(input_ids, max_length=128)
# decode output and print
print('Model output: ', t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True))

