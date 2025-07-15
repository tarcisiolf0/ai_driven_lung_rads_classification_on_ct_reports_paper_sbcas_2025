import json
import re


# Example usage
#input_file = 'llms/zero_shot/data/gemini/results_prompt_post_processed.txt'  
#output_file = 'llms/zero_shot/data/gemini/results_prompt.jsonl'

input_file = 'llms/few_shot/data/llama/results_prompt_post_processed_five_ex.txt'  
output_file = 'llms/few_shot/data/llama/results_prompt.jsonl'

# Step 1: Read the input file
with open(input_file, "r", encoding="utf-8") as f:
    data = f.read()

# Step 2: Extract JSON objects using regex
json_objects = re.findall(r"\{.*?\}", data, re.DOTALL)

# Step 3: Parse extracted JSON objects into a list of dictionaries
parsed_objects = [json.loads(obj) for obj in json_objects]

# Step 4: Group every 3 objects into sublists
grouped_data = [parsed_objects[i:i+3] for i in range(0, len(parsed_objects), 3)]

# Step 5: Write the output with each JSON array on a new line
with open(output_file, "w", encoding="utf-8") as f:
    for group in grouped_data:
        f.write(json.dumps(group, ensure_ascii=False) + "\n")  # Each list of 3 objects is a separate line

print(f"Conversion completed. Output saved as {output_file}.")


