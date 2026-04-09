"""
test_personal_caminos.py
Script para testing personal de Dijkstra y Bellman-Ford
con la visualización hecha por ChatGPT.
"""

import time
import networkx as nx # Más cómodo que usar llamadas a los helpers, la trabajamos en el HACKATHON.
import matplotlib.pyplot as plt

 # Importamos_todo
from Grafos1 import dijkstra, bellman_ford

# Función auxiliar para mostrar resultados intermedios en dijkstra
def mostrar_resultado_dijkstra(name, result, source, target):
    dist = result['distances'].get(target, None)
    path = result.get('path', [])
    n_steps = len(result.get('steps', []))
    print(f"\n{name}")
    print(f"  Distancia {source} → {target}: {dist}")
    print(f"  Camino: {' → '.join(path) if path else 'N/A'}")
    print(f"  Pasos registrados: {n_steps}")

    # Mostrar paso a paso
    print("\n  Pasos de ejecución:")
    for i, step in enumerate(result.get('steps', [])):
        current = step.get('current')
        visited = step.get('visited', set())
        distances = step.get('distances')
        print(f"    Paso {i+1}: nodo actual = {current}, visitados = {visited}")
        print(f"      Distancias: {distances}")


# Función auxiliar para mostrar resultados intermedios en Bellman-Ford
def mostrar_resultado_bellman_ford(name, result, source, target):
    dist = result['distances'].get(target, None)
    path = result.get('path', [])
    n_steps = len(result.get('steps', []))
    has_neg_cycle = result.get('has_negative_cycle', False)

    print(f"\n{name}")
    print(f"  Distancia {source} → {target}: {dist}")
    print(f"  Camino: {' → '.join(path) if path else 'N/A'}")
    print(f"  Pasos registrados: {n_steps}")
    print(f"  Ciclo negativo detectado: {has_neg_cycle}")

    # Mostrar paso a paso
    print("\n  Pasos de ejecución (relajación de arcos):")
    for i, step in enumerate(result.get('steps', [])):
        pass_num = step.get('pass', i+1)  # número de pasada
        relaxed_edges = step.get('relaxed_edges', [])  # lista de tuplas (u,v,distancia)
        distances = step.get('distances')
        converged = step.get('converged', False)

        print(f"    Pasada {pass_num}:")
        if relaxed_edges:
            for u, v, w, old_d, new_d in relaxed_edges:
                print(f"      Arco {u}->{v} peso={w}: {old_d} → {new_d}")
        else:
            print("      Ningún arco relajado en esta pasada")
        print(f"      Distancias actuales: {distances}")
        if converged:
            print("      [✓] Convergencia anticipada detectada, se puede parar")
            break

    if has_neg_cycle:
        print("  [✗] Se detectó un ciclo de peso negativo, resultados no confiables")


# Ejemplo en cuestión:
"""
graph = {
    'A': {'B': 2, 'C': 5},
    'B': {'C': 1, 'D': 4},
    'C': {'D': 1},
    'D': {}
}

source = 'A'
target = 'D'



graph = {
    'A': {'B': 3, 'C': 2},
    'B': {'C': 1, 'D': 6, 'E': 5},
    'C': {'D': 1, 'F': 4},
    'D': {'G': 2},
    'E': {'D': 1, 'F': 2},
    'F': {'G': 1},
    'G': {}
}

source = 'A'
target = 'G'
"""
graph = {
    'A': {'B': 2, 'C': 5},
    'B': {'C': 1, 'D': 2},
    'C': {'E': 3},
    'D': {'F': 1, 'G': 4},
    'E': {'D': -6, 'H': 2},  # Arco E→D genera ciclo negativo C→E→D→F→E
    'F': {'E': 1},
    'G': {'F': 2, 'I': 3},
    'H': {'G': -4},           # Arco H→G ayuda a crear otro ciclo negativo
    'I': {'J': 1},
    'J': {'H': 2}             # Arco J→H cierra un ciclo negativo
}

source = 'A'
target = 'J'


# Dijkstra
t0 = time.time()
res_dij = dijkstra(graph, source, target)
t_dij = time.time() - t0
mostrar_resultado_dijkstra("Dijkstra", res_dij, source, target)
print(f"  Tiempo de ejecución: {t_dij:.6f}s")

# Bellman-Ford
t1 = time.time()
res_bellman = bellman_ford(graph, source, target)
t_bell = time.time() - t1
mostrar_resultado_bellman_ford("Bellman-Ford", res_bellman, source, target)
print(f"  Tiempo de ejecución: {t_bell:.6f}s")



# Visualización del Grafo, by ChatGPT.
def dibujar_grafo(graph, path=None):
    G = nx.DiGraph()
    for u, vs in graph.items():
        for v, w in vs.items():
            G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='#a5d6a7', node_size=800)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.show()

# Visualizar camino Dijkstra y Bellman-Ford
dibujar_grafo(graph, path=res_dij.get('path'))
dibujar_grafo(graph, path=res_bellman.get('path'))