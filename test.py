import datatable as dt
from datatable import f
x = dt.Frame({'def':['a', 'b']})

y = x[:, [f.def]]