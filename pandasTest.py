import pandas as pd

se = pd.Series([1, 2, 3, 4, 5],name="aaa")
print(se)
custom_index = [1,2,3,4,5]
se = pd.Series([1, 2, 3, 4, 5], index=custom_index, name="aaa")
print(se)