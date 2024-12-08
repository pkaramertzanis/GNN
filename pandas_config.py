import pandas as pd

# pandas display options

# do not fold dataframes
pd.set_option('expand_frame_repr', False)
# maximum number of columns
pd.set_option("display.max_columns",100)
# maximum number of rows
pd.set_option("display.max_rows",500)
# precision of float numbers
pd.set_option("display.precision",3)
# maximum column width
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True