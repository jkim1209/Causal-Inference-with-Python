#%%
import networkx as nx
import matplotlib.pyplot as plt

model = nx.DiGraph([
    ("C", "A"),
    ("C", "B"),
    ("D", "A"),
    ("B", "E"),
    ("F", "E"),
    ("A", "G")
])

pos = nx.spring_layout(model, seed=23)
nx.draw(model, pos=pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=500)
plt.show()

#%%
# 충돌부 구조
print("---------\n충돌부 구조\n---------")
print("Are D and C dependent?")
print(not(nx.is_d_separator(model, {"D"}, {"C"}, set())), "\n")

print("Are D and C dependent given A?")
print(not(nx.is_d_separator(model, {"D"}, {"C"}, {"A"})), "\n")

print("Are D and C dependent given G?")
print(not(nx.is_d_separator(model, {"D"}, {"C"}, {"G"})), "\n")

#%%
# 사슬 구조
print("---------\n사슬 구조\n---------")
print("Are G and D dependent?")
print(not(nx.is_d_separator(model, {"G"}, {"D"}, set())), "\n")

print("Are G and D dependent given A?")
print(not(nx.is_d_separator(model, {"G"}, {"D"}, {"A"})), "\n")

#%%
# 분기 구조
print("---------\n분기 구조\n---------")
print("Are A and B dependent?")
print(not(nx.is_d_separator(model, {"A"}, {"B"}, set())), "\n")

print("Are A and B dependent given C?")
print(not(nx.is_d_separator(model, {"A"}, {"B"}, {"C"})), "\n")
