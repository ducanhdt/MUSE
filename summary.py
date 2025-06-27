import glob
import json
import os
import pandas as pd


dataset = 'knes'
# dataset = 'dbp15k'
if dataset == 'knes':
    lang_pairs = ["el-en", "el-es", "el-fr", "el-ja", "en-fr", "es-en", "es-fr", "ja-en", "ja-es", "ja-fr"]
    lang_pairs = [lang_pair.split('-') for lang_pair in lang_pairs]
    log_folder = "dumped"
    time_log = "time.txt"
    output_path = "summary_knes.csv"
elif dataset == 'dbp15k':
    lang_pairs = ["fr_en", "ja_en", "zh_en"]
    lang_pairs = [lang_pair.split('_') for lang_pair in lang_pairs]
    log_folder = "dumped_dbp15"
    time_log = "time_dbp15.txt"
    output_path = "summary_dbp15.csv"

data = []

time_df = pd.read_csv(time_log, sep='\t', header=0)

for method in ['unsupervised', 'supervised']:
    for src_lang, tgt_lang in lang_pairs:
        for train_size in [30, 50, 70, 90]:
            for embeding_type in ['mbert', 'xlm-r', 'me5']:
                output_file = f"{log_folder}/{method}_{src_lang}_{tgt_lang}_{embeding_type}_{train_size}/0/train.log"
                if not os.path.exists(output_file):
                    print("FAIL", output_file)
                    continue
                # read this log file and find the last line that contains __log__:
                #INFO - 06/07/25 05:26:51 - 0:00:03 - __log__:{"n_iter": 0, "precision_at_1-nn": 72.83715012722645, "precision_at_5-nn": 78.43511450381679, "precision_at_10-nn": 80.02544529262087, "precision_at_1-csls_knn_10": 77.03562340966921, "precision_at_5-csls_knn_10": 82.82442748091603, "precision_at_10-csls_knn_10": 84.16030534351145, "mean_cosine-nn-S2T-10000": 0.9173554182052612, "mean_cosine-csls_knn_10-S2T-10000": 0.9151573181152344}
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                    last_log_line = None
                    for line in lines:
                        if "__log__:" in line:
                            last_log_line = line.strip()
                    if not last_log_line:
                        print("No log found", output_file)
                # # find the last line that contains __log__ and extract the json part
                if "__log__:" in last_log_line:
                    json_part = last_log_line.split("__log__:")[-1].strip()
                    try:
                        log_data = json.loads(json_part)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e} for {output_file}")  
                    # add to dataframe
                    try:
                        if method == 'unsupervised' and train_size !=30:
                            training_time = time_df[(time_df['method'] == method) &
                                                    (time_df['src_lang'] == src_lang) &
                                                    (time_df['tgt_lang'] == tgt_lang) &
                                                    (time_df['embeding_type'] == embeding_type) &
                                                    (time_df['train_size'] == 30)]['time'].values[0]
                        else:
                            training_time = time_df[(time_df['method'] == method) &
                                                    (time_df['src_lang'] == src_lang) &
                                                    (time_df['tgt_lang'] == tgt_lang) &
                                                    (time_df['embeding_type'] == embeding_type) &
                                                    (time_df['train_size'] == train_size)]['time'].values[0]
                    except IndexError:
                        print(f"No training time found for {method}, {src_lang}, {tgt_lang}, {embeding_type}, {train_size}")
                        training_time = None
                    data.append({
                        'method': method,
                        'src_lang': src_lang,
                        'tgt_lang': tgt_lang,
                        'embeding_type': embeding_type,
                        'train_size': train_size,
                        'n_iter': log_data.get('n_iter', 0),
                        'precision_at_1-nn': log_data.get('precision_at_1-nn', None),
                        'precision_at_5-nn': log_data.get('precision_at_5-nn', None),
                        'precision_at_10-nn': log_data.get('precision_at_10-nn', None),
                        'precision_at_1-csls_knn_10': log_data.get('precision_at_1-csls_knn_10', None),
                        'precision_at_5-csls_knn_10': log_data.get('precision_at_5-csls_knn_10', None),
                        'precision_at_10-csls_knn_10': log_data.get('precision_at_10-csls_knn_10', None),
                        'train_time':  training_time,
                    })
                else:
                    print(f"No log found in {output_file}")
                    print("Last line:", last_log_line)

data = pd.DataFrame(data)
data.to_csv(output_path, index=False)
print(f"Summary saved to {output_path}")
