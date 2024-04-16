from InstructABSA.utils import T5Generator
from instructions import InstructionsHandler

# Set Global Values
instruct_handler = InstructionsHandler()

# Load instruction set 2 for ASPE
instruct_handler.load_instruction_set2()

print('Mode set to: Individual sample inference')


# Create T5 model object
model_checkpoint = "./Models/joint_task/kevinscariajoint_tk-instruct-base-def-pos-neg-neut-combined-robs_experiment"
t5_exp = T5Generator(model_checkpoint)
print("Model loaded from: ", model_checkpoint)
bos_instruction_id = instruct_handler.aspe['bos_instruct2']
eos_instruction = instruct_handler.aspe['eos_instruct']

# Get input from user
user_input = input("Enter sentence for inference: ")
# format and tokenize input
model_input = bos_instruction_id + user_input + eos_instruction
input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
# generate output
outputs = t5_exp.model.generate(input_ids, max_length=128)
# decode output and print
print('Model output: ', t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True))

