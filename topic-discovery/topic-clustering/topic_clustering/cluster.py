# Hierarical Grouping
import logging
from collections import defaultdict

import infomap
import networkx
import numpy as np

from networkx import community
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

class GraphCluster(object):

    def cluster(self, embeddings, top_k=64, min_corr=0.9):
        sim_matrix = cosine_similarity(embeddings)
        pruned_matrix, num_edges = self._prune_matrix(sim_matrix, top_k, min_corr)
        if num_edges != 0:
            try:
                communities = self._detect_community(pruned_matrix)
            except:
                res = {i: [i] for i in range(len(embeddings))}
                return res
        else:
            res = {i: [i] for i in range(len(embeddings))}
            return res
        res = self._to_standard_grouping_format(communities, embeddings.shape[0])
        return res
            
    def _prune_matrix(self, sim_matrix, top_k, min_corr):
        pruned_matrix = {}
        num_edges = 0
        for node_i, row in enumerate(sim_matrix):
            neighbors = {}
            corrs = [(node, v) for node, v in enumerate(row)]
            for node, corr in corrs:
                if corr > min_corr:
                    neighbors[node] = corr
                    num_edges += 1
            pruned_matrix[node_i] = neighbors
        return pruned_matrix, num_edges
    
    def _detect_community(self, pruned_matrix):
        im = infomap.Infomap("--two-level --verbose --silent")
        for node_i in pruned_matrix:
            for node_j in pruned_matrix[node_i]:
                im.add_link(node_i, node_j)
        im.run()
        return im
    
    def _to_standard_grouping_format(self, communities, size):
        """Format: Dict{Key=str: Group_ID, Values=List: Embedding_IDs}"""
        res = defaultdict(list)
        node_in_group = {i: False for i in range(size)}
        max_module_id = 0
        
        for node_id, module_id in communities.modules:
            res[module_id].append(node_id)
            node_in_group[node_id] = True
            max_module_id = max(max_module_id, module_id)

        # Note, infomap would drop the communities whose size is one.
        # so we have to append it mannually.
        for node_id, in_group in node_in_group.items():
            if not in_group:
                max_module_id += 1
                res[max_module_id] = [node_id]
        return res

class BucketCluster(object):        
    
    def __init__(self, num_bucket_in_node=32, max_depth=12,
                 bucket_size_for_graph_cluster=30000, edge_sim_threshold=0.6, 
                 scalable_cluster_method=MiniBatchKMeans):
        self.num_bucket_in_node = num_bucket_in_node
        self.max_depth = max_depth
        self.bucket_size_for_graph_cluster = bucket_size_for_graph_cluster
        self.edge_sim_threshold = edge_sim_threshold
        self.scalable_cluster_method = scalable_cluster_method
        self.depth = 0
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        self.scalable_cluster_method = scalable_cluster_method # it should follow the interface as KMeans.
    
    def iterate(self, embeddings_idx, embeddings_values):
        # bind the embedding_idx with embeddings
        embeddings = [(idx, val) for idx, val in zip(embeddings_idx, embeddings_values)]
        group_mapping = self._iterate(embeddings)
        result = self._traverse_group_mapping(group_mapping)
        return result
    
    def _iterate(self, embeddings):
        print(f"In depth: {self.depth}. Grouping on embeddings size: {len(embeddings)}, gonna divide it into {self.num_bucket_in_node} buckets.")
        group_mapping = self._scalable_cluster(embeddings)
        for group_id in group_mapping:
            subset = self._get_subset(embeddings, group_mapping[group_id])
            bucket_size = len(group_mapping[group_id])           
            
            # For a small enough subset,
            # we can get the mean-sim and determine how to merge it using graph cluster.
            if bucket_size < self.bucket_size_for_graph_cluster:
                group_mapping[group_id] = self._graph_cluster(subset)
                        
            # Otherwise, we have to split the node and keep divide it to small enough buckets.
            elif self.depth < self.max_depth:
                self.depth += 1
                group_mapping[group_id] = self._iterate(subset)
                self.depth -= 1
            else:
                # Meet the max-depth, early-stopping.
                group_mapping[group_id] = {0: [i for i in range(len(subset))]}
        return group_mapping
    
    def _traverse_group_mapping(self, group_mapping):
        result = {}
        stack = [group_mapping]
        
        while len(stack):
            current_node = stack.pop()
            for group_values in current_node.values():
                if type(group_values) == list:
                    result[len(result)] = group_values
                else:
                    stack.append(group_values)
        return result
    
    def _scalable_cluster(self, embeddings):
        e_values = [v[1] for v in embeddings]
        
        # apply sklearn clustering.
        cluster = self.scalable_cluster_method(n_clusters=self.num_bucket_in_node)
        res = cluster.fit_predict(e_values)
        
        # format the result.
        group_mapping = self._format_sklearn_clustering_results(res)
        return group_mapping
    
    def _graph_cluster(self, embeddings):
        e_values = [v[1] for v in embeddings]
        e_indices = [v[0] for v in embeddings]
        graph_cluster = GraphCluster()
        res = graph_cluster.cluster(e_values, min_corr=self.edge_sim_threshold)
        
        # align the subset_id to e_indices
        aligned_res = {}
        for group_id, subset_ids in res.items():
            aligned_res[group_id] = [e_indices[v] for v in subset_ids]
        return aligned_res
           
    def _get_subset(self, embeddings, indices):
        return [embeddings[idx] for idx in indices]

    def _get_similarity(self, subset):
        sim = 0
        subset_size = len(subset)
        sim_matrix = cosine_similarity(subset)
        sim_matrix = np.triu(sim_matrix, 1) # get the upper triangle of the sim matrix.
        sim = np.sum(sim_matrix)
        normed_sim = sim * 2 / (subset_size * subset_size)
        return normed_sim    
    
    def _format_sklearn_clustering_results(self, sklearn_cluster_output):
        """
        Args:
            - sklearn_cluster_output: list[int], 
        Return
            - group_mapping: dict{key=group_id: str, val=group_values: list[int]}
        """
        group_mapping = defaultdict(list)
        for embedding_id, group_id in enumerate(sklearn_cluster_output):
            group_mapping[group_id].append(embedding_id)
        return group_mapping
    

class ClusteringHelper(object):
    
    def __init__(self, max_group_element_nums=40000,
                 group_saturated_bound=0.75):
        self.max_group_element_nums = max_group_element_nums
        self.group_saturated_bound = group_saturated_bound
        
        self.logger = logging.getLogger("ClusteringHelper|")
        self.logger.setLevel(logging.INFO)
        
    def get_candidates(self, embeddings: list, groups: dict, prev_candidates: dict):
        # step 1.
        self.logger.info("Checking group status.")
        group_saturated_flags = self._check_group_similarity(embeddings, 
                                                             groups,
                                                             prev_candidates)
        
        # step 2.
        self.logger.info("Picking candidates for clustering.")
        candidate_ids, candidate_to_related_ids = self._pick_candidates(groups, group_saturated_flags)
        return candidate_ids, candidate_to_related_ids
    
    def merge_back_candidates(self, candidate_to_related_ids, cur_groups: dict):
        for group_name in cur_groups:
            augmented_res = []
            for idx in cur_groups[group_name]:
                if idx in candidate_to_related_ids:
                    # the candidate behold the whole group.
                    augmented_res += candidate_to_related_ids[idx]
            cur_groups[group_name] += augmented_res
            # set -> list -> set to remove the duplicate candidates
            # which is already in candidate_to_related_ids[idx].
            cur_groups[group_name] = list(set(cur_groups[group_name]))
        return cur_groups

    def _check_group_similarity(self, 
                                embeddings, 
                                groups, 
                                prev_candidates=None):
        flags = []
        base = 0
        for idx, embedding_ids in enumerate(groups.values()):
            # a heurstics hack to prevent from calculating pairwise distances on a huge group.
            if len(embedding_ids) > self.max_group_element_nums:
                flags.append(False)
            else:
                # we should only evaluate on the indices 
                # which has been chosed as candidates in the previous grouping.
                if prev_candidates is not None:
                    embedding_ids = [v for v in embedding_ids if v in prev_candidates]
                if len(embedding_ids) == 0:
                    # it's because there is no element in this group had been chosen as a candidate
                    # in previous grouping. Therefore, this group is saturated.
                    flags.append(True)
                else:
                    similarity = self._get_subset_similarity(embeddings, embedding_ids)
                    if similarity > self.group_saturated_bound:
                        flags.append(True)
                    else:
                        flags.append(False)
        return flags
    
    def _pick_candidates(self, groups, group_saturated_flags):
        candidate_ids = []
        candidate_saturated_flags = {}
        candidate_to_related_ids = {}
        for group_id, is_saturated in zip(groups, group_saturated_flags):
            embedding_ids = list(groups[group_id])
            if is_saturated:
                # we always pick the first element to represent the whole group
                candidate_ids.append(embedding_ids[0])
                candidate_to_related_ids[embedding_ids[0]] = embedding_ids
            else:
                candidate_ids += embedding_ids
        return candidate_ids, candidate_to_related_ids
            
    def _get_subset(self, embeddings, ids):
        subset = [embeddings[i] for i in ids]
        return subset
    
    def _get_subset_similarity(self, embeddings, ids):
        sim = 0
        subset = self._get_subset(embeddings, ids)
        subset_size = len(subset)
        sim_matrix = cosine_similarity(subset)
        sim_matrix = np.triu(sim_matrix, 1) # get the upper triangle of the sim matrix.
        sim = np.sum(sim_matrix)
        normed_sim = sim * 2 / (subset_size * subset_size)
        return normed_sim
