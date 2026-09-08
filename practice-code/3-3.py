#%%
import networkx as nx

consultancy_model_served = nx.DiGraph([
    ("profits_prev_6m", "profits_next_6m"),
    ("profits_prev_6m", "consultancy"),
    # ("consultancy", "profits_next_6m")    #  인과관계 제거
])

not(nx.is_d_separator(consultancy_model_served, {"profits_prev_6m"}, {"profits_next_6m"}, set()))