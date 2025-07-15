from classification import classify_nodules
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == '__main__':

    # Carregar os DataFrames Zero Shot
    #df_results_deep_seek = pd.read_csv("llms/zero_shot/data/deep_seek/results_prompt_structured_post_processed.csv")
    #df_results_gemini = pd.read_csv("llms/zero_shot/data/gemini/results_prompt_structured_post_processed.csv")
    #df_results_gpt = pd.read_csv("llms/zero_shot/data/gpt/results_prompt_structured_post_processed.csv")
    #df_results_llama = pd.read_csv("llms/zero_shot/data/llama/results_prompt_structured_post_processed.csv")


    # Carregar os DataFrames Few Shot
    df_results_deep_seek = pd.read_csv(r"llms\few_shot\data\deep_seek\results_prompt_five_ex_structured_post_processed.csv")
    df_results_gemini = pd.read_csv(r"llms\few_shot\data\gemini\results_prompt_five_ex_structured_post_processed.csv")
    df_results_gpt = pd.read_csv(r"llms\few_shot\data\gpt\results_prompt_five_ex_structured_post_processed.csv")
    df_results_llama = pd.read_csv(r"llms\few_shot\data\llama\results_prompt_five_ex_structured_post_processed.csv")

    #df_structured_data = pd.read_excel('doc_similarity/data/structured_data_for_lung_rads.ods')
    #lung_rads_column = df_structured_data[df_structured_data['Lung-RADS'].notna()]
    #df_structured_data_filtered = df_structured_data[~df_structured_data["Laudo"].astype(str).str.contains(r'\.(1|2|3)$', regex=True)]
    #lung_rads_column = df_structured_data_filtered['Lung-RADS'].dropna()
    #list_lung_rads = lung_rads_column.to_list()
    
    test_df = pd.read_csv(r'doc_similarity\data\test.csv')
    list_lung_rads = test_df['Lung-RADS'].to_list()

    for i in range(len(list_lung_rads)):
        item = list_lung_rads[i]
        if item == 0.0: list_lung_rads[i] = "0"
        elif item == 1.0: list_lung_rads[i] = "1"
        elif item == 2.0: list_lung_rads[i] = "2"
        elif item == 3.0: list_lung_rads[i] = "3"
        elif item == 4.0: list_lung_rads[i] = "4A"
        elif item == 5.0: list_lung_rads[i] = "4B"
        elif item == 6.0: list_lung_rads[i] = "4X"
    

    df_results_deep_seek.insert(6 ,"Lung-RADS", list_lung_rads)
    df_results_gemini.insert(6 ,"Lung-RADS", list_lung_rads)
    df_results_gpt.insert(6 ,"Lung-RADS", list_lung_rads)
    df_results_llama.insert(6 ,"Lung-RADS", list_lung_rads)

    # Classificar os nódulos nos DataFrames
    df_results_deep_seek['Lung-RADS-Pred'] = classify_nodules(df_results_deep_seek)
    df_results_gemini['Lung-RADS-Pred'] = classify_nodules(df_results_gemini)
    df_results_gpt['Lung-RADS-Pred'] = classify_nodules(df_results_gpt)
    df_results_llama['Lung-RADS-Pred'] = classify_nodules(df_results_llama)

    # Exibir o DataFrame com as classificações
    
    
    y_true = list_lung_rads
    y_pred_deep_seek = df_results_deep_seek['Lung-RADS-Pred'].tolist()
    y_pred_gemini = df_results_gemini['Lung-RADS-Pred'].tolist()
    y_pred_gpt = df_results_gpt['Lung-RADS-Pred'].tolist()
    y_pred_llama = df_results_llama['Lung-RADS-Pred'].tolist()

    y_true_str = [str(x) for x in y_true]
    y_pred_deep_seek = [str(item) for item in y_pred_deep_seek]
    y_pred_gemini = [str(item) for item in y_pred_gemini]
    y_pred_gpt = [str(item) for item in y_pred_gpt]
    y_pred_llama = [str(item) for item in y_pred_llama]


    metrics_deep_seek = classification_report(y_true_str, y_pred_deep_seek, zero_division=0.0, output_dict=True)
    metrics_gemini = classification_report(y_true_str, y_pred_gemini, zero_division=0.0, output_dict=True)
    metrics_gpt = classification_report(y_true_str, y_pred_gpt, zero_division=0.0, output_dict=True)
    metrics_llama = classification_report(y_true_str, y_pred_llama, zero_division=0.0, output_dict=True)

    df_metrics_deep_seek = pd.DataFrame(data=metrics_deep_seek).transpose()
    df_metrics_gemini = pd.DataFrame(data=metrics_gemini).transpose()
    df_metrics_gpt = pd.DataFrame(data=metrics_gpt).transpose()
    df_metrics_llama = pd.DataFrame(data=metrics_deep_seek).transpose()

    df_metrics_deep_seek.to_csv(r'lung_rads\data\few_shot\lung_rads_metrics_deep_seek.csv')
    df_metrics_gemini.to_csv(r'lung_rads\data\few_shot\lung_rads_metrics_gemini.csv')
    df_metrics_gpt.to_csv(r'lung_rads\data\few_shot\lung_rads_metrics_gpt.csv')
    df_metrics_llama.to_csv(r'lung_rads\data\few_shot\lung_rads_metrics_llama.csv')

    df_results_deep_seek.to_csv(r'lung_rads\data\few_shot\lung_rads_predictions_deep_seek.csv', index=False)
    df_results_gemini.to_csv(r'lung_rads\data\few_shot\lung_rads_predictions_gemini.csv', index=False)
    df_results_gpt.to_csv(r'lung_rads\data\few_shot\lung_rads_predictions_gpt.csv', index=False)
    df_results_llama.to_csv(r'lung_rads\data\few_shot\lung_rads_predictions_llama.csv', index=False)
