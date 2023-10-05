from dask import dataframe as dask_df

# Put the path to your SOEP panel file "pl.csv" here
main_panel_file = "HERE"

df = dask_df.read_csv(main_panel_file)
my_df = df[["pid", "syear", "plh0171", "plh0173"]]
print(my_df)

my_df.to_csv('pl_filtered.csv', index=False, single_file=True)
