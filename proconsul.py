"""
This code was adapted from the DIAMOnD algorithm code aviliable at:
https://github.com/dinaghiassian/DIAMOnD
"""

import argparse
import csv
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F


# =============================================================================
def print_usage():

    print(' ')
    print('        usage: python3 proconsul.py --network_file --seed_file --n --alpha(optional) --outfile_name(optional) --n_rounds(optional) --temp(optional) --top_p(optional) --top_k(optional)')
    print('        -----------------------------------------------------------------')
    print('        network_file : The edgelist must be provided as any delimiter-separated')
    print('                       table. Make sure the delimiter does not exit in gene IDs')
    print('                       and is consistent across the file.')
    print('                       The first two columns of the table will be')
    print('                       interpreted as an interaction gene1 <==> gene2')
    print('        seed_file    : table containing the seed genes (if table contains')
    print('                       more than one column they must be tab-separated;')
    print('                       the first column will be used only)')
    print('        n            : desired number of genes to predict, 200 is a reasonable')
    print('                       starting point.')
    print('        alpha        : an integer representing weight of the seeds,default')
    print('                       value is set to 1')
    print('        outfile_name : results will be saved under this file name')
    print('                       by default the outfile_name is set to "first_n_added_nodes_weight_alpha.txt"')
    print('        n_rounds     : How many different rounds PROCONSUL will do to reduce statistical fluctuation.')
    print('                       (default: 10)' )
    print('        temp         : Temperature value for the softmax function.')
    print('                       (default: 1.0)')
    print('        top_p        : Probability threshold value for nucleus sampling. If 0 no nucleus sampling.')
    print('                       (default: 0.0)')
    print('        top_k        : Length of the p-values subset for Top-K sampling. If 0 no top-k sampling.')
    print('                       (default: 0)')
    print(' ')


# =============================================================================

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithms and validation')
    parser.add_argument('--network_file', type=str,
                    help='Path to the edgelist to be used for building the network. The edgelist must be provided as any delimiter-separated table. Make sure the delimiter does not exit in gene IDs and is consistent across the file.')
    parser.add_argument('--seed_file', type=str,
                    help='table containing the seed genes (if table contains more than one column they must be tab-separated;the first column will be used only)')
    parser.add_argument('--n', type=int,
                    help='desired number of genes to predict, 200 is a reasonable starting point.')
    parser.add_argument('--alpha', type=int, default=1,
                    help='an integer representing weight of the seeds (default: 1)')
    parser.add_argument('--outfile_name', type=str, default="default",
                    help='results will be saved under this file name (default: "proconsul_n_predicted_genes_temp_t(top_p_top_k_).txt")')
    parser.add_argument('--n_rounds', type=int, default=10,
                    help='How many different rounds PROCONSUL will do to reduce statistical fluctuation. (default: 10)')
    parser.add_argument('--temp', type=float, default=1.0,
                    help='Temperature value for the PROCONSUL softmax function. (default: 1.0)')
    parser.add_argument('--top_p', type=float, default=0.0,
                    help='Probability threshold value for PROCONSUL nucleus sampling. If 0 no nucleus sampling. (default: 0.0)')
    parser.add_argument('--top_k', type=int, default=0,
                    help='Length of the pvalues subset for Top-K sampling. If 0 no top-k sampling. (default: 0)')
    return parser.parse_args()


# =============================================================================
def read_files(network_file, seed_file):
    """
    Reads the network and the list of seed genes from external files.
    * The edgelist must be provided as a tab-separated table. The
    first two columns of the table will be interpreted as an
    interaction gene1 <==> gene2
    * The seed genes mus be provided as a table. If the table has more
    than one column, they must be tab-separated. The first column will
    be used only.
    * Lines that start with '#' will be ignored in both cases
    """

    sniffer = csv.Sniffer()
    line_delimiter = None
    for line in open(network_file, 'r'):
        if line[0] == '#':
            continue
        else:
            dialect = sniffer.sniff(line)
            line_delimiter = dialect.delimiter
            break
    if line_delimiter == None:
        print
        'network_file format not correct'
        sys.exit(0)

    # read the network:
    G = nx.Graph()
    for line in open(network_file, 'r'):
        # lines starting with '#' will be ignored
        if line[0] == '#':
            continue
        # The first two columns in the line will be interpreted as an
        # interaction gene1 <=> gene2
        # line_data   = line.strip().split('\t')
        line_data = line.strip().split(line_delimiter)
        node1 = line_data[0]
        node2 = line_data[1]
        G.add_edge(node1, node2)

    # read the seed genes:
    seed_genes = set()
    for line in open(seed_file, 'r'):
        # lines starting with '#' will be ignored
        if line[0] == '#':
            continue
        # the first column in the line will be interpreted as a seed
        # gene:
        line_data = line.strip().split('\t')
        seed_gene = line_data[0]
        seed_genes.add(seed_gene)

    return G, seed_genes

def read_terminal_input(args):
    """
    Reads the arguments passed by command line.
    """

    network_file    = args.network_file
    seed_file       = args.seed_file
    n               = args.n
    alpha           = args.alpha
    outfile_name    = args.outfile_name
    n_rounds        = args.n_rounds
    temp            = args.temp
    top_p           = args.top_p
    top_k           = args.top_k

    # Check arguments
    if n < 1:
        print(f"ERROR: The number of genes to predict must be greater or equal 1.")
        print_usage()
        sys.exit(1)
    
    if alpha < 0:
        print(f"ERROR: alpha must be greater or equal 0.")
        print_usage
        sys.exit(1)

    if n_rounds < 1:
        print(f"ERROR: The number of PROCONSUL rounds must be greater or equal 1.")
        print_usage()
        sys.exit(1)

    if temp < 0:
        print(f"ERROR: The temperature must be greater or equal 0.")
        print_usage()
        sys.exit(1)
    
    if temp == 0:
        # Sustitue 0 with a very small number to avoid nan values
        temp = 1e-40
    
    if top_p < 0:
        print(f"ERROR: The probability threshold for nucleus sampling (top-p) must be greater or equal 0.")
        print_usage()
        sys.exit(1)

    if top_k < 0:
        print(f"ERROR: The number of p-values subset for top-k sampling must be greater or equal 0.")
        print_usage()
        sys.exit(1)
    
    # Generate the default outfile
    if outfile_name == "default":
        outfile_name = f"proconsul_{n}_predicted_genes_{n_rounds}_rounds_temp_{temp}.txt"

        # Add top-p and top-k if they are greater than 0
        if top_p > 0.0:
            outfile_name = outfile_name.replace(".txt", f"_top_p_{top_p}.txt")
        
        if top_k > 0:
            outfile_name = outfile_name.replace(".txt", f"_top_k_{top_k}.txt")
    
    # Read network and seed files
    G_original, seed_genes = read_files(network_file, seed_file)

    return G_original, seed_genes, n, alpha, outfile_name, n_rounds, temp, top_p, top_k


# ================================================================================
def compute_all_gamma_ln(N):
    """
    precomputes all logarithmic gammas
    """
    gamma_ln = {}
    for i in range(1, N + 1):
        gamma_ln[i] = scipy.special.gammaln(i)

    return gamma_ln


# =============================================================================
def logchoose(n, k, gamma_ln):
    if n - k + 1 <= 0:
        return scipy.infty
    lgn1 = gamma_ln[n + 1]
    lgk1 = gamma_ln[k + 1]
    lgnk1 = gamma_ln[n - k + 1]
    return lgn1 - [lgnk1 + lgk1]


# =============================================================================
def gauss_hypergeom(x, r, b, n, gamma_ln):
    return np.exp(logchoose(r, x, gamma_ln) +
                  logchoose(b, n - x, gamma_ln) -
                  logchoose(r + b, n, gamma_ln))


# =============================================================================
def pvalue(kb, k, N, s, gamma_ln):
    """
    -------------------------------------------------------------------
    Computes the p-value for a node that has kb out of k links to
    seeds, given that there's a total of s seeds in a network of N nodes.

    p-val = \sum_{n=kb}^{k} HypergemetricPDF(n,k,N,s)
    -------------------------------------------------------------------
    """
    p = 0.0
    for n in range(kb, k + 1):
        if n > s:
            break
        prob = gauss_hypergeom(n, s, N - s, k, gamma_ln)
        # print prob
        p += prob

    if p > 1:
        return [1]
    else:
        return p

    # =============================================================================


def get_neighbors_and_degrees(G):
    neighbors, all_degrees = {}, {}
    for node in G.nodes():
        nn = set(G.neighbors(node))
        neighbors[node] = nn
        all_degrees[node] = G.degree(node)

    return neighbors, all_degrees


# =============================================================================
# Reduce number of calculations
# =============================================================================
def reduce_not_in_cluster_nodes(all_degrees, neighbors, G, not_in_cluster, cluster_nodes, alpha):
    reduced_not_in_cluster = {}
    kb2k = defaultdict(dict)
    for node in not_in_cluster:

        k = all_degrees[node]
        kb = 0
        # Going through all neighbors and counting the number of module neighbors
        for neighbor in neighbors[node]:
            if neighbor in cluster_nodes:
                kb += 1

        # adding wights to the the edges connected to seeds
        k += (alpha - 1) * kb
        kb += (alpha - 1) * kb
        kb2k[kb][k] = node

    # Going to choose the node with largest kb, given k
    k2kb = defaultdict(dict)
    for kb, k2node in kb2k.items():
        min_k = min(k2node.keys())
        node = k2node[min_k]
        k2kb[min_k][kb] = node

    for k, kb2node in k2kb.items():
        max_kb = max(kb2node.keys())
        node = kb2node[max_kb]
        reduced_not_in_cluster[node] = (max_kb, k)

    return reduced_not_in_cluster


# This implementation cames from: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """

    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits



# ======================================================================================
#   C O R E    A L G O R I T H M
# ======================================================================================
def proconsul_iteration_of_first_X_nodes(G, S, X, alpha, temp=1.0, top_k=0, top_p=0):
    """
    Parameters:
    ----------
    - G:     graph
    - S:     seeds
    - X:     the number of iterations, i.e only the first X gened will be
             pulled in
    - alpha: seeds weight
    - temp:  the temperaure value for the softmax function
    - top_k: number of subset of p-values for top-k sampling
    - top_p: prob threshold for top-p sampling

    Returns:
    --------

    - added_nodes: ordered list of nodes in the order by which they
      are agglomerated. Each entry has 4 info:
      * name : dito
      * k    : degree of the node
      * kb   : number of +1 neighbors
      * p    : p-value at agglomeration
    """

    N = G.number_of_nodes()

    added_nodes = []

    # ------------------------------------------------------------------
    # Setting up dictionaries with all neighbor lists
    # and all degrees
    # ------------------------------------------------------------------
    neighbors, all_degrees = get_neighbors_and_degrees(G)

    # ------------------------------------------------------------------
    # Setting up initial set of nodes in cluster
    # ------------------------------------------------------------------

    cluster_nodes = set(S)
    not_in_cluster = set()
    s0 = len(cluster_nodes)

    s0 += (alpha - 1) * s0
    N += (alpha - 1) * s0

    # ------------------------------------------------------------------
    # precompute the logarithmic gamma functions
    # ------------------------------------------------------------------
    gamma_ln = compute_all_gamma_ln(N + 1)

    # ------------------------------------------------------------------
    # Setting initial set of nodes not in cluster
    # ------------------------------------------------------------------
    for node in cluster_nodes:
        not_in_cluster |= neighbors[node]
    not_in_cluster -= cluster_nodes

    # ------------------------------------------------------------------
    #
    # M A I N     L O O P
    #
    # ------------------------------------------------------------------

    all_p = {}

    while len(added_nodes) < X:
        # ------------------------------------------------------------------
        #
        # Going through all nodes that are not in the cluster yet and
        # record k, kb and p
        #
        # ------------------------------------------------------------------

        info = {}

        next_node = 'nix'
        reduced_not_in_cluster = reduce_not_in_cluster_nodes(all_degrees,
                                                             neighbors, G,
                                                             not_in_cluster,
                                                             cluster_nodes, alpha)

        probable_next_nodes = []
        p_values = []

        for node, kbk in reduced_not_in_cluster.items():
            # Getting the p-value of this kb,k
            # combination and save it in all_p, so computing it only once!
            kb, k = kbk
            try:
                p = all_p[(k, kb, s0)]
            except KeyError:
                p = pvalue(kb, k, N, s0, gamma_ln)
                all_p[(k, kb, s0)] = p

            info[node] = (k, kb, p)

            # Save the neighbour in the probable next nodes array and its p-value
            probable_next_nodes.append(node)
            p_values.append(p[0])



        # ---------------------------------------------------------------------
        # Get the negative logarithm of pvalues, use them as reference point
        # to create a probability distribution and use it to draw the next node
        # ---------------------------------------------------------------------

        # Cast the p-values list to a Tensor
        p_values = torch.tensor(p_values, dtype=torch.float64)

        # Get the negative logarithm of the p-values to use
        log_p_values = -torch.log(p_values)

        # Scale by the temperature
        log_p_values /= temp

        # Top-K and Top-P filtering
        log_p_values = top_k_top_p_filtering(log_p_values, top_k=top_k, top_p=top_p)

        # Sample from the filtered distribution
        probabilities = F.softmax(log_p_values, dim=-1)

        # Check on probabilities
        if True in torch.isnan(probabilities):
            print("ERROR: found nan value")
            sys.exit(1)

        # Finally draw the next node
        next_node = probable_next_nodes[torch.multinomial(probabilities, 1)]


        # ---------------------------------------------------------------------
        # Adding the sorted node to the list of agglomerated nodes
        # ---------------------------------------------------------------------
        added_nodes.append((next_node,
                            info[next_node][0],
                            info[next_node][1],
                            info[next_node][2]))

        # Updating the list of cluster nodes and s0
        cluster_nodes.add(next_node)
        s0 = len(cluster_nodes)
        not_in_cluster |= (neighbors[next_node] - cluster_nodes)
        not_in_cluster.remove(next_node)

    return added_nodes


# ===================================================================================
#  P R O C O N S U L
#  U s e   t h i s   f u n c   t o   c a l l   i t   f r o m   a   t h i r d   a p p  
# ===================================================================================

def PROCONSUL(G_original, seed_genes, max_number_of_added_nodes, alpha, outfile=None, n_rounds=10, temp=1.0, top_k=0, top_p=0.0):

    # 1. throwing away the seed genes that are not in the network
    all_genes_in_network = set(G_original.nodes())
    seed_genes = set(seed_genes)
    disease_genes = seed_genes & all_genes_in_network

    if len(disease_genes) != len(seed_genes):
        print("PROCONSUL(): ignoring %s of %s seed genes that are not in the network" % (
            len(seed_genes - all_genes_in_network), len(seed_genes)))

    # 2. agglomeration algorithm.
    node_ranks = {}
    for i in range(n_rounds):
        print(f"PROCONSUL(): Round {i+1}/{n_rounds}")
        added_nodes = proconsul_iteration_of_first_X_nodes(G_original,
                                                           disease_genes,
                                                           max_number_of_added_nodes,
                                                           alpha,
                                                           temp=temp,
                                                           top_k=top_k,
                                                           top_p=top_p)

        # Assign rank value to the node
        for pos, node in enumerate(added_nodes):
            node_number = node[0]

            if node_number not in node_ranks:
                node_ranks[node_number] = len(added_nodes) - pos # if pos = 0 => rank = 100 - 0 = 100
            else:
                node_ranks[node_number] += len(added_nodes) - pos


    # Average the rank of each node by the total number of PROCONSUL iterations
    for key in node_ranks.keys():
        node_ranks[key] /= n_rounds

    # Sort the dictionary in descendig order wrt the rank values
    sorted_nodes = sorted(node_ranks.items(), key=lambda x: x[1], reverse=True)

    # 3. saving the results
    with open(outfile, 'w') as fout:

        fout.write('\t'.join(['rank', 'node', 'rank_score']) + '\n')
        rank = 0
        for sn in sorted_nodes:
        	
            if rank >= max_number_of_added_nodes:
                break
        		
            rank += 1
            node = sn[0]
            rank_score = sn[1]

            fout.write('\t'.join(map(str, ([rank, node, rank_score]))) + '\n')

    return sorted_nodes[:max_number_of_added_nodes]


# =============================================================
#  M A I N:  T o   r u n   i t   b y   c o m m a n d   l i n e
# =============================================================

if __name__ == '__main__':

    # Read input
    args = parse_args()
    G_original, seed_genes, max_number_of_added_nodes, alpha, outfile_name, n_rounds, temp, top_p, top_k = read_terminal_input(args)


    # run PROCONSUL
    added_nodes = PROCONSUL(G_original,
                            seed_genes,
                            max_number_of_added_nodes,
                            alpha,
                            outfile=outfile_name,
                            n_rounds=n_rounds,
                            temp=temp,
                            top_p=top_p,
                            top_k=top_k)

    print("\n results have been saved to '%s' \n" % outfile_name)
