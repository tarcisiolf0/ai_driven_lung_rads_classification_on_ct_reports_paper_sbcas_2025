import json
from tqdm import tqdm
import csv

def read_data(file_name):
    return json.load(open(file_name, encoding="utf-8"))

def read_idx(file_name):
    print("reading ...")
    example_idx = []
    with open(file_name, "r") as file:
        for line in file:
            example_idx.append(json.loads(line.strip()))
    return example_idx

def read_similarity_results(file_name):
    similarity_results = {}
    with open(file_name, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            report_id = int(float(row['report_idx']))
            similar_5 = json.loads(row['five_most_similars'])
            similarity_results[report_id] = {'five_most_similars': similar_5}
    return similarity_results


def construct_prompt(train_data, train_tables, test_data, similarity_results, example_num):
    print("prompt ...")

    id_to_list_idx = {item['id']: i for i, item in enumerate(train_data)}

    def get_example(index):
        example_prompt = ""
        for idx_ in similarity_results[index]['five_most_similars'][:example_num]:
            idx_ = int(idx_)
            real_idx = id_to_list_idx.get(idx_)
            int_id = int(float(train_data[real_idx]["id"]))
            text = train_data[real_idx]["text"]
            table = train_tables[real_idx]

            # Ad ID
            example_prompt += f"O laudo exemplo: {text}\n"
            example_prompt += f"O laudo exemplo com a tabela preenchida: {table}\n"
        return example_prompt

    results = []
    inputs = []

    for item_idx in tqdm(range(len(test_data))):
        
        item_ = test_data[item_idx]
        int_id = item_['id']
        text = item_['text']

        
        # PROMPT 1
        prompt = """Você receberá um laudo de tomografia computadorizada do tórax. Sua tarefa é extrair informações relevantes dos nódulos pulmonares e armazenar essas informações em um dicionário JSON que deve possuir o seguinte formato:

{
"O nódulo tem atenuação sólida ou em partes moles?" : "resposta",
"O nódulo tem atenuação em vidro fosco?" : "resposta",
"O nódulo tem borda espiculada ou irregular ou mal definida?" : "resposta",
"O nódulo é calcificado?" : "resposta",
"Localização do nódulo" : "resposta",
"Tamanho do nódulo" : "resposta"
}

A seguir são descritos alguns requisitos para extração:
1. Por favor extraia informações estruturadas para o nódulo pulmonar mencionado no laudo para preencher a tabela.
2. Todas as perguntas devem ser respondidas com "Sim" ou "Não, exceto as perguntas da "Localização do nódulo" e "Tamanho do nódulo".
3. Sólido ou partes moles e vidro fosco, são mutuamente exclusivas. Apenas uma das duas perguntas pode ser respondida com "Sim".
4. Se o laudo não contiver informações relevantes relacionadas a uma pergunta específica ou se você não souber responder à pergunta, por favor preencha com "Não" a resposta dessa pergunta. 
5. A pergunta do tamanho do nódulo deve ser respondida apenas com números e unidade de medida.
6. Caso haja mais de um nódulo descrito no laudo, retornar apenas a tabela de perguntas do nódulo com maior tamanho.

Aqui são descritos alguns pontos de conhecimento médico prévio para sua referência
1. Imagem ovalar hiperdensa deve ser considerada como nódulo pulmonar calcificado.
2. Micronódulo é um nódulo no pulmão com menos de 3 milímetros de diâmetro. Nesse contexto devido as suas pequenas dimensões não estamos interessados em extrair suas características.
3. Massa pulmonar é qualquer área de opacificação pulmonar que mede mais de 3 cm. Nesse contexto devido as suas grandes dimensões não estamos interessados em extrair suas características.

Abaixo estão alguns exemplos de laudos com as tabelas preenchidas, e você deve fazer as mesmas previsões que os exemplos.
"""
        prompt += get_example(index=item_['id']) # Usando o ID do item atual para buscar exemplos similares
        prompt += '\n'

        input = f"Dado o laudo: {text}\nRetornar a tabela do laudo preenchida no formato JSON.\n\n"

        inputs.append(input)
        results.append(prompt)
        
    return inputs, results


if __name__ == '__main__':
    train_samples = read_data(r"C:\Users\tarcisio.ferreira\Desktop\Eu\mestrado\Lung_RADS_SBCAS_2025\doc_similarity\data\train_samples.json")
    test_samples = read_data(r"C:\Users\tarcisio.ferreira\Desktop\Eu\mestrado\Lung_RADS_SBCAS_2025\doc_similarity\data\test_samples.json")
    train_tables = read_data(r"C:\Users\tarcisio.ferreira\Desktop\Eu\mestrado\Lung_RADS_SBCAS_2025\doc_similarity\data\train_tabels.json")
    similarity_results = read_similarity_results(r"C:\Users\tarcisio.ferreira\Desktop\Eu\mestrado\Lung_RADS_SBCAS_2025\doc_similarity\data\similarity_results.csv") # Lendo os resultados de similaridade

    inputs, prompts = construct_prompt(train_data=train_samples, train_tables=train_tables, test_data=test_samples, 
                                       similarity_results=similarity_results, example_num=5)

    
    with open(r"llms\few_shot\data\inputs.txt", encoding="utf-8", mode="w") as txt_file:
        for line in inputs:
            txt_file.write("".join(line) + "\n") # works with any number of elements in a line
    

    with open(r"llms\few_shot\data\prompt_five_ex.txt", encoding="utf-8", mode="w") as txt_file:
        for line in prompts:
            txt_file.write("".join(line) + "\n") # works with any number of elements in a line
    