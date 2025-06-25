import glob
import json
import os
import pandas as pd
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

data = []

for method in ['unsupervised', 'supervised']:
    for link_path in os.listdir("/home/cs.aau.dk/qz83md/MUSE/Knes_dataset/data/seed_alignlinks/"):
        src_lang, tgt_lang = link_path.split('.')[0].split('-')
        for train_size in [30, 50, 70, 90]:
            for embeding_type in ['mbert', 'xml']:
                output_file = f"dumped/{method}_{src_lang}_{tgt_lang}_{embeding_type}_{train_size}/0/train.log"
                if not os.path.exists(output_file):
                    print("FAIL", output_file)
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
                    })
                else:
                    print(f"No log found in {output_file}")
                    print("Last line:", last_log_line)

data = pd.DataFrame(data)
data.to_csv("summary.csv", index=False)
print("Summary saved to summary.csv")
# push to a public google sheet
# Set up the authentication
# scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

# # Path to your Google API credentials JSON file
# credentials_file = 'google_credentials.json'

# try:
#     # Authenticate
#     creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
#     client = gspread.authorize(creds)
    
#     # Open the Google Sheet
#     sheet_url = "https://docs.google.com/spreadsheets/d/1LSQEBoUWtOq5xREbOObST40YA70z7171CoX97mYoiqU/edit?usp=sharing"
#     spreadsheet = client.open_by_url(sheet_url)
#     worksheet = spreadsheet.get_worksheet(0)
    
#     # Clear existing data and update with the DataFrame
#     worksheet.clear()
    
#     # Convert DataFrame to list format and update
#     header = data.columns.tolist()
#     values = data.values.tolist()
#     all_data = [header] + values
#     worksheet.update('A1', all_data)
    
#     print("Data successfully pushed to Google Sheet")
# except Exception as e:
#     print(f"Failed to push data to Google Sheet: {str(e)}")
#     print("Make sure you have set up the Google API credentials correctly")
