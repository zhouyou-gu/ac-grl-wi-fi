import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
# G = nx.complete_graph(10)
p = np.random.choice([0. , 1.], (2,2),p=[0.8, 0.2])

print(p)
G = nx.from_numpy_matrix(p,create_using=nx.MultiDiGraph())
for i in range(G.number_of_nodes()):
    print(list(G.neighbors(i)))
nx.draw(G)
plt.show()

