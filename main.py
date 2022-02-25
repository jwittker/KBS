import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import (bipartite, centrality, community, clustering, link_prediction,
                                 distance_measures)
from networkx.drawing.layout import bipartite_layout
from networkx.classes.function import degree_histogram, density, subgraph, common_neighbors

import queue, heapq

import clingo

class Node :
    def __init__(self, n, t):
        self.m_name = n
        self.m_type = t

class Edge:
    def __init__(self, n1, n2):
        self.to_node = n1
        self.from_node = n2

# == COMMUNITY ANALYTICS ===========================================================================
def intra_edges(c):
    """
    the number of intra_edges of community C
    """
    pass

def inter_edges(c):
    """
    the number of inter_edges of community C
    """
    pass

def inter_degree(u):
    """
    inter-degree for a node u in community C
    """
    pass

def inverse_average_odf(c):
    """
    inverse average out-degree faction
    compares the number of inter-edges to the number of all edges of a community C
    and averages this for the wohle community by considering the fraction for each individual node
    """
    pass

def segregation_index(c):
    """
    compares the number of expected inter-edges to the number of observed inter-edges
    normalized by the expectation
    """
    pass

def modularity(graph, comm):
    """
    modularity of a graph clustering with k communities focuses on the number of edges within a community
    compares that with the xpected such number given a null-model
    """
    return community.modularity(graph, comm)

def modularity_contribution(c):
    """
    modularity contribution of a single community c in a local context (subgraph)
    """
    pass

# == COMMUNITY DETECTION ===========================================================================
# COMODO
def comodo(graph, max_pattern_size, max_pattern_length, min_community_size):
    db = transform(graph, db)
    cpt = create_cpt(db, min_community_size)
    top_k = queue.PriorityQueue(max_pattern_length) # priority queue
    comodo_mine(top_k)

    return top_k

def transform(graph):
    return graph.edges

def create_cpt(dbmin_community_size):
    cpt = []

    return cpt

def comodo_mine(cpt, pattern, top_k, max_pattern_size, max_pattern_length, min_community_size):
    """
    branch & bouond algorithm based on an exhaustive subgroup discovery approach
    applies extended SD-Map* method
    """
    COM = {}
    min_q = heapq.nsmallest(1)

    for basic_pattern in cpt:
        p = createRefinement(pattern, basic_pattern)
        COM[basic_pattern] = p

        if p > min_community_size:
            if quality(p) > min_q:
                top_k.put(p)
                min_q = top_k.get()[0]

    if len(pattern) + 1 < max_pattern_length:
        refinements  = sortBasicPatternsByOptimisticEstimateDescending()
        for basic_pattern in refinements:
            if optimisticEstimate(COM[basic_pattern]) >= min_q:
                ccpt = getConditionalCPT(basic_pattern, cpt, min_q)
                comodo(ccpt, COM[basic_pattern], top_k)

def createRefinement(pattern, basic_pattern):
    pass

def quality(p):
    pass

def sortBasicPatternsByOptimisticEstimateDescending():
    pass

def optimisticEstimate(com):
    pass

def getConditionalCPT(basic_pattern, cpt, min_q):
    pass

# MINERLSD
def minerLSD(graph):
    """
    integrates abstract closed pattern mining with efficient pruning approaches
    """
    pass

# == CLUSTERING COEFFICIENTS =======================================================================
def clustering_coefficient(graph, nodes=None):
    return clustering(graph, nodes)

# == SIMPLE GRAPHS ANALYTICS =======================================================================
def diameter(graph):
    return distance_measures.diameter(graph)

# == CENTRALITIES ==================================================================================
def betweenness(graph):
    return centrality.betweenness_centrality(graph)

def closeness(graph, u=None):
    return centrality.closeness_centrality(graph, u)

def eigen(graph):
    try:
        result = centrality.eigenvector_centrality(graph)
    except Exception as e:
        print("ERROR IN EIGENVECTOR CENTRALITY", e)
        return False

    return result

# == NODE SIMILARITIES =============================================================================
def neighborhood_sim(graph, u, v):
    count = 1
    if graph.degree(u) == graph.degree(v):
        for n in graph.neighbors(u):
            if n == v:
                continue
            for m in graph.neighbors(v):
                if m == u:
                    continue
                if n == m:
                    count += 1

    return (count == graph.degree(u) and count == graph.degree(v))

def distance_sim(graph, u, v):
    pass

def structued_sim(graph, u, v):
    pass

# == LINK PREDICTION ===============================================================================
def cold_start_link_prediction(graph):
    """
    Solves the cold start link prediction problem for an attributed graph
    """
    # setup clingo program
    control = clingo.Control()

    node_attributes = ["work_base", "group_affiliation"]
    # node_attributes = ["work_base", "group_affiliation", "publisher"]
    node_node = ["relatives"]

    attribute_dict = {}
    # preprocess attributes by mapping them to numbers
    number = -1 # choose negative numbers, since nodes have positive numbers
    for node in graph.nodes:
        for attr in node_attributes:
            all_nodes_attributes = nx.get_node_attributes(graph, attr)
            if not all_nodes_attributes[node] in attribute_dict and not all_nodes_attributes[node] == "-":
                attribute_dict[all_nodes_attributes[node]] = number
                number -= 1

    node_dict = {}
    node_number = 0
    # map node names to numbers and setup node facts
    for node in graph.nodes:
        if not node in node_dict:
            node_dict[node] = node_number
            node_number += 1

        node_to_add = "node(" + str(node_dict[node]) + ")."
        control.add("cold_start", [], node_to_add)

        # setup bipartite attribute graph
        node_attr_to_add = "node_attr(" + str(node_dict[node]) + ")."
        control.add("cold_start", [], node_attr_to_add)

        for attr in node_attributes:
            node_attrs = nx.get_node_attributes(graph, attr)
            # skip empty attribute
            if node_attrs[node] == "-":
                continue
            attribute_of_node = attribute_dict[node_attrs[node]]
            attr_to_add = "a(" + str(attribute_of_node) + ")."
            control.add("cold_start", [], attr_to_add)

            edge_node_attr_to_add = "edge_attr(" + str(node_dict[node]) + "," + str(attribute_of_node) + ")."
            control.add("cold_start", [], edge_node_attr_to_add)

        # node_node_info = nx.get_node_attributes(graph, "relatives")

    # setup edges
    for edge in graph.edges:
        edge_to_add = "edge(" + str(node_dict[edge[0]]) + "," + str(node_dict[edge[1]]) + ")."
        control.add("cold_start", [], edge_to_add)

    # setup rules
    # undirected edges
    control.add("cold_start", [], "edge(Y,X) :- edge(X,Y).")
    control.add("cold_start", [], "edge_attr(Y,X) :- edge_attr(X,Y).")

    # is X common neighbor of Y and Z without an edge between them
    control.add("cold_start", [], "c(X,Y,Z) :- edge(X,Y), edge(X,Z), not edge(Y,Z), Y!=Z.")
    control.add("cold_start", [], "c_attr(X,Y,Z) :- edge_attr(X,Y), edge_attr(X,Z), not edge_attr(Y,Z), Y!=Z.")

    # if two nodes have one common attribute_neighbor, predict link
    control.add("cold_start", [], "cn_lp(Y,Z) :- node_attr(Y), node_attr(Z), not edge_attr(Y,Z), 1=#count{X:c_attr(X,Y,Z)}.")

    control.ground([("cold_start", [])])

    def on_model(m):
        """
        Retrieve the link prediction atoms ("cn_lp") and create new edges based off of them
        """
        count = 0
        node_keys = list(node_dict.keys())
        node_values = list(node_dict.values())
        for atom in m.symbols(atoms=True):
            if "cn_lp" in str(atom):
                left_node = int(str(atom).split("(")[1].split(",")[0])
                right_node = int(str(atom).split(",")[1].split(")")[0])
                graph.add_edge(node_keys[node_values.index(left_node)], node_keys[node_values.index(right_node)])
                count += 1
        print(int(count / 2), " edges added.")

    control.solve(on_model=on_model)
    control.cleanup()


def structured_link_prediction(graph):
    """
    Solves one iteration of link prediction of a graph based on its structure
    """
    # setup clingo program
    control = clingo.Control()

    node_dict = {}
    node_number = 0
    # map node names to numbers and setup node facts
    for node in graph.nodes:
        if not node in node_dict:
            node_dict[node] = node_number
            node_number += 1

        node_to_add = "node(" + str(node_dict[node]) + ")."
        control.add("struct_link_pred", [], node_to_add)

    # setup edges
    for edge in graph.edges:
        edge_to_add = "edge(" + str(node_dict[edge[0]]) + "," + str(node_dict[edge[1]]) + ")."
        control.add("struct_link_pred", [], edge_to_add)

    # setup rules
    # undirected edges
    control.add("struct_link_pred", [], "edge(Y,X) :- edge(X,Y).")

    # is X common neighbor of Y and Z without an edge between them
    control.add("struct_link_pred", [], "c(X,Y,Z) :- edge(X,Y), edge(X,Z), not edge(Y,Z), Y!=Z.")

    # if two nodes have two common neighbors, predict link
    control.add("struct_link_pred", [], "cn_lp(Y,Z) :- node(Y), node(Z), not edge(Y,Z), 2=#count{X:c(X,Y,Z)}.")

    control.ground([("struct_link_pred", [])])

    def on_model(m):
        """
        Retrieve the link prediction atoms ("cn_lp") and create new edges based off of them
        """
        count = 0
        # node_keys = list(node_dict.keys())
        # node_values = list(node_dict.values())
        for atom in m.symbols(atoms=True):
            if "cn_lp" in str(atom):
                # left_node = int(str(atom).split("(")[1].split(",")[0])
                # right_node = int(str(atom).split(",")[1].split(")")[0])
                # graph.add_edge(node_keys[node_values.index(left_node)], node_keys[node_values.index(right_node)])
                count += 1
        print(int(count / 2), " edges added.")

    control.solve(on_model=on_model)
    control.cleanup()


# == LOAD DATA & PREPROCESSING =====================================================================
# (csv have been slightly adapted to be separated by ';' since some names include ',' already)

def load_graphs():
    # FIRST GRAPH - MarvelUniverse
    # (node, type)
    data_nodes = pd.read_csv("data/MarvelUniverse/nodes.csv", sep=";").reset_index()

    # (hero, comic) - probably not needed, see paper for explanation
    # data_comic_edges = pd.read_csv("data/MarvelUniverse/edges.csv", sep=";")

    # (hero1, hero2)
    data_marvel_edges = pd.read_csv("data/MarvelUniverse/hero-network.csv", sep=";")


    # SECOND GRAPH - SUPER HERO DATASET
    super_nodes = pd.read_csv("data/Heroes/data.csv")

    # drop all information/attributes not contributing to graph structure (decrease data size)
    attributes_to_drop = ["powerstats__intelligence", "powerstats__strength", "powerstats__speed"
                        ,"powerstats__durability", "powerstats__power", "powerstats__combat"
                        ,"biography__place-of-birth" ,"biography__first-appearance"
                        ,"appearance__gender", "appearance__race"
                        ,"appearance__height__001", "appearance__height__002"
                        ,"appearance__weight__001", "appearance__weight__002"
                        ,"appearance__eye-color", "appearance__hair-color", "work__occupation"]
    for i in range(2, 21):
        if i <= 9:
            attr_to_drop = "biography__aliases__00" + str(i)
        else:
            attr_to_drop = "biography__aliases__0" + str(i)
        attributes_to_drop.append(attr_to_drop)

    for col in attributes_to_drop:
        del super_nodes[col]

    super_nodes["name"] = super_nodes["name"].str.upper()
    super_nodes["biography__full-name"] = super_nodes["biography__full-name"].str.upper()
    super_nodes["connections__relatives"] = super_nodes["connections__relatives"].str.upper()

    marvel_nodes = data_nodes["node"][data_nodes["type"] == "hero"].unique().tolist()

    # == CREATE GRAPHS =================================================================================
    marvel_graph = nx.Graph()
    marvel_graph.add_nodes_from(marvel_nodes)

    marvel_edges = data_marvel_edges.values.tolist()
    marvel_edges = [tuple(x) for x in marvel_edges]
    marvel_graph.add_edges_from(marvel_edges)

    super_graph = nx.Graph()
    # add nodes with selection of attributes
    for idx, row in super_nodes.iterrows():
        super_graph.add_node(row["name"], work_base=row["work__base"],
                                          full_name=["biography__full-name"],
                                          alter_egos=["biography__alter-egos"],
                                          alias=row["biography__aliases__001"],
                                          publisher=row["biography__publisher"],
                                          alignment=row["biography__alignment"],
                                          group_affiliation=row["connections__group-affiliation"],
                                          relatives=row["connections__relatives"])

    return (marvel_graph, super_graph)

# == COLD START LINK PREDICTION ====================================================================
marvel_graph, super_graph = load_graphs()

# # super hero graph
# cold_start_link_prediction(super_graph)

# structured_link_prediction(super_graph)

# marvel graph
print("Old number of edges: ", marvel_graph.number_of_edges())
structured_link_prediction(marvel_graph)
