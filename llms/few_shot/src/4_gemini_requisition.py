import google.generativeai as genai
import time
import process_files as pf
from dotenv import load_dotenv
import os

def gemini_req(inputs, prompts):
    results = []
    total_exec_time = 0

    genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
    system_message = "Você é um experiente radiologista em analisar laudos de TC do tórax para realizar a classificação Lung-RADS dos nódulos pulmonares."

    for i in range(len(inputs)):     
        input = inputs[i]
        input_results = []

        prompt = prompts[i]

        for j in range(3):
            model = genai.GenerativeModel('gemini-2.0-flash', generation_config={"response_mime_type": "application/json", "temperature" : 0})

            start_time = time.time()   

            final_prompt = ''
            #final_prompt = input+"\n\n"+prompt
            final_prompt = system_message+"\n\n"+prompt+"\n\n"+input
            response = model.generate_content(final_prompt)

            ## Tempo 
            end_time = time.time()
            exec_time = end_time - start_time
            total_exec_time += exec_time
            
            print(f"Request {j+1} for input {i+1}:\n {response.text}") # Indicate which request it is
            pf.print_execution_stats(i, exec_time, total_exec_time)
            input_results.append(response.text)
        
            # Sleep to avoid hitting the rate limit
            time.sleep(5)
            
        results.append(input_results)
        
    return results


if __name__ == '__main__':
    load_dotenv()
    # Prompt - 5 Examples
    inputs = pf.read_input_file("few_shot\data\inputs.txt")
    inputs = pf.pre_process_input_file(inputs)

    prompts_five_ex = pf.read_input_file("few_shot\data\prompt_five_ex.txt")
    prompts_five_ex = pf.pre_process_input_file(prompts_five_ex)

    results_one_five_ex = gemini_req(inputs, prompts_five_ex)
    pf.write_output_file(r"few_shot\data\gemini\results_prompt_five_ex.txt", results_one_five_ex)
