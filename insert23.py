import os
import csv
import pandas as pd



CT_AGV1 = current.CT_AGV1.value
CT_P_AGV1 = current.CT_P_AGV1.value

Fetch_AGV1_Log = current.Fetch_AGV1_Log.value
Fetch_AGV1_Log_F = current.Fetch_AGV1_Log_F.value

WIP = current.WIP.value
Products_Finished = current.Products_Finished_AGV1.value

NLP_D = current.NLP_D.value



logfile = "kpi_log_3.csv"
header = ["WIP", "Products_Finished", "CT_WS1", "WS1", "WS1_F", "WS2", "WS2_F", "CT_WS2", "CT_WS3", "WS3", "WS3_F", "CT_P_WS1", "CT_P_WS2", "CT_P_WS3",
          "CT_AGV1", "AGV1", "AGV1_F", "CT_AGV2", "AGV2", "AGV2_F", "CT_P_AGV1", "CT_P_AGV2", "NLP_D"]

row_dict = {
    "WIP": WIP,
    "Products_Finished": Products_Finished,
    "CT_AGV1": CT_AGV1,
    "CT_P_AGV1": CT_P_AGV1,
    "AGV1": Fetch_AGV1_Log,
    "AGV1_F": Fetch_AGV1_Log_F,
    "NLP_D": NLP_D,
    # Only fill in *new* values here. Leave others blank or don't specify.
}

def update_kpi_log(logfile, header, row_dict):
    # 1. Read or create empty DataFrame
    if os.path.exists(logfile) and os.path.getsize(logfile) > 0:
        df = pd.read_csv(logfile)
    else:
        df = pd.DataFrame(columns=header)

    # 2. Prepare types so matching works (especially if numbers/strings get mixed)
    for col in ['WIP', 'Products_Finished']:
        if col in df.columns and col in row_dict:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            row_dict[col] = float(row_dict[col]) # or int, depends on your input

    # 3. Find matching row
    mask = (
        (df['WIP'] == row_dict['WIP']) &
        (df['Products_Finished'] == row_dict['Products_Finished']) &
        (df['NLP_D'] == row_dict['NLP_D'])
    )

    # 4. Update or append
    if mask.any():
        idx = df[mask].index[0]
        for k, v in row_dict.items():
            df.at[idx, k] = v
    else:
        # Fill row for all headers
        full_row = {col: row_dict.get(col, '') for col in header}
        df = pd.concat([df, pd.DataFrame([full_row], columns=header)], ignore_index=True)

    # 5. Write back, always keeping the header
    df.to_csv(logfile, index=False)

update_kpi_log(logfile, header, row_dict)

print("------ BEFORE ------")
if os.path.exists(logfile): print(open(logfile).read())
print("------ ADD/UPDATE ------")
print(row_dict)
print("------ AFTER ------")
df = pd.read_csv(logfile)
print(df)
