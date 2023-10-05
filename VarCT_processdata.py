import pandas
import numpy as npy
import time

# Panel data file
file = 'data_processed/pl_filtered.csv'
df = pandas.read_csv(file)

# Identify subjects by pid
pid_column = df["pid"]
all_pidsUnique = pid_column.drop_duplicates()

# Parameters to possibly normalize data if not done before
mean = 0
norm = 1


def processPanelData():
    print("Reading panel data from " + file + " ...")
    start = time.time()
    years = []
    values = []
    # theMax will become the time horizon = largest year of participation found
    theMax = 0

    # Iterate each pid
    for pid_row in all_pidsUnique:
        years_perPid = []
        values_perPid = []
        df_perPid = df[df['pid'] == pid_row]
        # Iterate rows per pid
        for index, perPid_row in df_perPid.iterrows():
            # Normalize participation years to start from 0
            # SOEP study started 1984
            theYear_perPid = perPid_row['syear']-1984
            theMax = npy.maximum(npy.max(theYear_perPid), theMax)
            years_perPid.append(theYear_perPid)
            values_perPid.append([(perPid_row['plh0171']-mean)/norm, (perPid_row['plh0173']-mean)/norm])
        # Discard participants with only one data point
        if len(years_perPid) > 1:
            years.append(years_perPid)
            values.append(values_perPid)

    end = time.time()
    print("... done in {:1.2f}s!".format(end-start))
    return years, values, theMax


# Validator to check for years consistency, run VarCT_processdata directly
if __name__ == "__main__":
    years, values = processPanelData()
    print(years)
