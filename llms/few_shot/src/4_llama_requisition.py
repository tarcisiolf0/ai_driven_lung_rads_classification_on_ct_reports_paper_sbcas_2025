import time
from together import Together
import process_files as pf
import os
from dotenv import load_dotenv

def llama_req(inputs, prompts):
    results = []
    total_exec_time = 0

    system_message = "Você é um experiente radiologista em analisar laudos de TC do tórax para realizar a classificação Lung-RADS dos nódulos pulmonares."

    for i in range(len(inputs)):
        
        prompt = prompts[i]
        input = inputs[i]
        input_results = []

        content = prompt+"\n\n"+input
        #print(content)
        for j in range(3):
            client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

            start_time = time.time()

            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": content}
                ],
                temperature=0
            )

            text_response = response.choices[0].message.content
            ## Tempo 
            end_time = time.time()
            exec_time = end_time - start_time
            total_exec_time += exec_time
            
            print(f"Request {j+1} for input {i+1}:\n {text_response}")
            pf.print_execution_stats(i, exec_time, total_exec_time)
            input_results.append(response.choices[0].message.content)

        results.append(input_results)
        
    return results
    

if __name__ == '__main__':
    # Prompt - 5 Examples
    load_dotenv()

    inputs = pf.read_input_file("llms/few_shot/data/inputs.txt")
    inputs = pf.pre_process_input_file(inputs)

    prompts_one_five_ex = pf.read_input_file("llms/few_shot/data/prompt_five_ex.txt")
    prompts_one_five_ex = pf.pre_process_input_file(prompts_one_five_ex)

    results_one_five_ex = llama_req(inputs, prompts_one_five_ex)
    pf.write_output_file("llms/few_shot/data/llama/results_prompt_five_ex.txt", results_one_five_ex)
   
