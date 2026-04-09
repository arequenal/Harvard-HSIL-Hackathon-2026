
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, List, Tuple, Optional
import math

# Modificación, se la pasé a ChatGPT.
import matplotlib
matplotlib.use('TkAgg')  # o 'Qt5Agg' si Tk no funciona
import matplotlib.pyplot as plt

Graph = {
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

def reconstruct_path(
    predecessors: Dict[str, Optional[str]],
    source: str,
    target: str
) -> Optional[List[str]]:
    """
    Reconstruye el camino de source a target a partir del diccionario
    de predecesores.

    Parámetros
    ----------
    predecessors : dict  {nodo: nodo_predecesor o None}
    source       : nodo origen
    target       : nodo destino

    Devuelve
    --------
    Lista de nodos [source, ..., target] o None si target no es alcanzable.
    """
    if target not in predecessors or (predecessors[target] is None and target != source):
        # target inalcanzable — comprobar si es el propio source
        if target == source:
            return [source]
        return None

    path = []
    node = target
    visited = set()
    while node is not None:
        if node in visited:
            return None  # ciclo en predecesores
        visited.add(node)
        path.append(node)
        node = predecessors.get(node)

    path.reverse()
    if path[0] != source:
        return None
    return path


# ═══════════════════════════════════════════════════════════════════
# 2. UTILIDADES DE GRAFO
# ═══════════════════════════════════════════════════════════════════

def get_all_nodes(graph: Graph) -> List[str]:
    """Devuelve todos los nodos del grafo (incluyendo destinos sin salida)."""
    nodes = set(graph.keys())
    for u in graph:
        for v in graph[u]:
            nodes.add(v)
    return sorted(nodes)


def get_all_edges(graph: Graph) -> List[Tuple[str, str, float]]:
    """Devuelve lista de (origen, destino, peso) para todas las aristas."""
    edges = []
    for u in graph:
        for v, w in graph[u].items():
            edges.append((u, v, w))
    return edges


def dijkstra(
    graph: Dict[str, Dict[str, float]],
    source: str,
    target: str,
) -> dict:
    """
    Algoritmo de Dijkstra (conjuntos P y T).

    Parámetros
    ----------
    graph  : diccionario de adyacencia {nodo: {vecino: peso, ...}, ...}
    source : nodo origen (str)
    target : nodo destino (str)

    Devuelve
    --------
    dict con claves:
        'distances'    : dict {str: float} — distancia mínima desde source
        'predecessors' : dict {str: str o None} — predecesor de cada nodo
        'path'         : list[str] — camino [source, ..., target], o None
        'steps'        : list[dict] — un dict por cada vértice que pasa a P:
                           'current'      : str  — vértice recién confirmado
                           'distances'    : dict — copia de dist en este instante
                           'predecessors' : dict — copia de prev
                           'visited'      : set  — copia de P (permanentes)

    Notas
    -----
    - Inicialización: P = {source}, dist[source] = 0.
      Para cada j ≠ source: dist[j] = peso del arco (source, j) si existe, +∞ si no.
    - En cada iteración: k = min dist[j] para j en T, pasar k a P,
      actualizar vecinos de k que estén en T.
    - get_all_nodes(graph) devuelve la lista de todos los vértices del grafo.
    - IMPORTANTE: hacer copia (dict(...), set(...)) al registrar cada step.
    """

    nodes = get_all_nodes(graph)

    # ── inicialización ──
    distances = {node: float('inf') for node in nodes}
    predecessors = {node: None for node in nodes}

    distances[source] = 0

    # Inicialización según enunciado
    for node in nodes:
        if node != source:
            if node in graph[source]:
                distances[node] = graph[source][node]
                predecessors[node] = source

    P = {source}
    T = set(nodes) - P

    steps = []

    # ── bucle principal ──
    while T:
        # Elegir nodo con menor distancia en T
        k = min(T, key=lambda node: distances[node])

        P.add(k)
        T.remove(k)

        # Registrar paso (IMPORTANTE: copias)
        steps.append({
            'current': k,
            'distances': dict(distances),
            'predecessors': dict(predecessors),
            'visited': set(P)
        })

        # Parada anticipada
        if k == target:
            break

        # Relajación
        for neighbor, weight in graph.get(k, {}).items():
            if neighbor in T:
                if distances[k] + weight < distances[neighbor]:
                    distances[neighbor] = distances[k] + weight
                    predecessors[neighbor] = k

    # Reconstrucción del camino
    path = reconstruct_path(predecessors, source, target)

    return {
        'distances': distances,
        'predecessors': predecessors,
        'path': path,
        'steps': steps
    }


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN 2: BELLMAN-FORD
# ─────────────────────────────────────────────────────────────────

def bellman_ford(
    graph: Dict[str, Dict[str, float]],
    source: str,
    target: str,
) -> dict:
    """
    Algoritmo de Bellman-Ford con detección de circuitos negativos.

    Parámetros
    ----------
    graph  : diccionario de adyacencia {nodo: {vecino: peso, ...}, ...}
    source : nodo origen (str)
    target : nodo destino (str), puede ser None para single-source

    Devuelve
    --------
    dict con claves:
        'distances'          : dict {str: float} — distancia mínima desde source
        'predecessors'       : dict {str: str o None} — predecesor de cada nodo
        'path'               : list[str] — camino [source, ..., target], o None
        'steps'              : list[dict] — un dict por cada pasada completa:
                                 'iteration'     : int  — número de pasada (1, 2, ...)
                                 'distances'     : dict — copia de dist tras la pasada
                                 'predecessors'  : dict — copia de prev
                                 'updated_edges' : list[tuple[str,str]] — arcos mejorados
        'has_negative_cycle' : bool — True si se detecta un circuito negativo

    Notas
    -----
    - Inicialización: dist[source] = 0.
      Para cada j ≠ source: dist[j] = peso del arco (source, j) si existe, +∞ si no.
      Esto cubre los caminos de 1 arco.
    - Después: |V|-2 pasadas recorriendo TODOS los arcos del grafo.
    - Convergencia anticipada: si ninguna distancia mejora en una pasada, parar.
    - Detección de circuito negativo: una pasada extra tras las regulares.
    - get_all_nodes(graph) devuelve la lista de todos los vértices del grafo.
    - IMPORTANTE: hacer copia (dict(...)) al registrar cada step.
    """

    nodes = get_all_nodes(graph)
    distances = {node: float('inf') for node in nodes}
    predecessors = {node: None for node in nodes}

    distances[source] = 0

    # Inicialización de vecinos del source
    for node in nodes:
        if node != source and node in graph[source]:
            distances[node] = graph[source][node]
            predecessors[node] = source

    steps = []

    n = len(nodes)
    has_negative_cycle = False

    # ── pasadas principales (n-1) ──
    for t in range(n - 1):
        distances_copy = dict(distances)
        change = False

        for u in graph:
            for v, w in graph[u].items():
                if distances[u] + w < distances_copy[v]:
                    distances_copy[v] = distances[u] + w
                    predecessors[v] = u
                    change = True

        distances = distances_copy

        steps.append({
            'current_pass': t + 1,
            'distances': dict(distances),
            'predecessors': dict(predecessors),
        })

        if not change:
            break  # No hay cambios → terminar temprano

    # ── detección de ciclos negativos ──
    for u in graph:
        for v, w in graph[u].items():
            if distances[u] + w < distances[v]:
                has_negative_cycle = True
                break
        if has_negative_cycle:
            break

    path = None
    if not has_negative_cycle:
        path = reconstruct_path(predecessors, source, target)

    return {
        'distances': distances,
        'predecessors': predecessors,
        'path': path,
        'steps': steps,
        'has_negative_cycle': has_negative_cycle
    }
