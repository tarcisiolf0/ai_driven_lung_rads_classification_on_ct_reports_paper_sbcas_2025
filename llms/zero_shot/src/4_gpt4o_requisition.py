import time
from openai import OpenAI
import process_files as pf
import os
from dotenv import load_dotenv

def gpt_req(inputs, prompt):
    results = []
    total_exec_time = 0

    system_message = "Você é um experiente radiologista em analisar laudos de TC do tórax para realizar a classificação Lung-RADS dos nódulos pulmonares."

    for i in range(len(inputs)):
        input = inputs[i]      
        input_results = []

        content = prompt+"\n"+input
        for j in range(3):
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

            start_time = time.time()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": content}
                ],
                seed=42,
                temperature=0
            )

            text_response = response.choices[0].message.content
            #print(text_response)
            
            ## Tempo 
            end_time = time.time()
            exec_time = end_time - start_time
            total_exec_time += exec_time
            
            print(f"Request {j+1} for input {i+1}: {text_response}")
            pf.print_execution_stats(i, exec_time, total_exec_time)
            input_results.append(response.choices[0].message.content)
        
        results.append(input_results)

    return results
    

if __name__ == '__main__':
   # Prompt
    load_dotenv()

    inputs = pf.read_input_file("llms/zero_shot/data/inputs.txt")
    prompt = pf.read_prompt_file("llms/zero_shot/data/prompt.txt")

    inputs = pf.pre_process_input_file(inputs)

    results = gpt_req(inputs, prompt)
    pf.write_output_file("llms/zero_shot/data/gpt/results_prompt.txt", results)

