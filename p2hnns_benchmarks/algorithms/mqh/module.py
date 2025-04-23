import numpy
import b_tree
import mh
import pymqh as mqh
from ..base.module import BaseANN

class BT_MQH(BaseANN):
    def __init__(self, metric, max_leaf_size=16, M2=16, level=4, m_level=1, m_num=64):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("BT_MQH doesn't support metric %s" % metric)
        
        # MQH parameters
        self._metric = metric
        self._M2 = M2
        self._level = level
        self._m_level = m_level
        self._m_num = m_num
        self._l0 = 3  # Default parameter
        self._delta = 0.5  # Default parameter
        self._flag = 0  # Default parameter
        self._initial_candidates = 1  # Default parameter
        
        # BTree parameters
        self._max_leaf_size = max_leaf_size
        self._c = 10.0
        self._tree_candidates = b_tree.BTree()
        
        # Query parameters
        self._candidates = 100  # Default
        self._initial_topk = 100  # Default

    def index(self, X):
        self._data = X.astype(numpy.float32)

        if self._metric == "angular":
            # Normalize for angular distance
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]

        # Build the MQH index for refinement
        n, d = self._data.shape
        self._mqh = mqh.MQH(d, self._M2, self._level, self._m_level, self._m_num)
        self._mqh.build_index(self._data)
        
        # For BTree, add ones column after any normalization
        ones_column = numpy.ones((self._data.shape[0], 1), dtype=numpy.float32)
        self._candidate_data = numpy.hstack((self._data, ones_column))

        n, d = self._candidate_data.shape
        data_array = numpy.ascontiguousarray(self._candidate_data.ravel())
        self._tree_candidates.preprocess(n, d, self._max_leaf_size, data_array)

    def set_query_arguments(self, candidates=100, initial_topk=100, l0=3, delta=0.5, flag=0, initial_candidates=1):
        self._candidates = candidates
        self._initial_topk = initial_topk
        self._l0 = l0
        self._delta = delta
        self._flag = flag
        self._initial_candidates = initial_candidates

    def query(self, q, b, n):
        """
        Query using BTree for candidate generation and MQH for refinement.
        
        Parameters:
        -----------
        q : numpy array
            Normal vector of the hyperplane
        b : float
            Bias term of the hyperplane equation <q, x> + b = 0
        n : int
            Number of nearest neighbors to return
            
        Returns:
        --------
        indices : list
            Indices of the n nearest neighbors to the hyperplane
        """
        # Calculate query norm
        qnorm = numpy.linalg.norm(q)

        # Prepare query vector for BTree
        if self._metric == "angular":
            q_normalized = q / qnorm
            b_normalized = b / qnorm
            q_to_pass = numpy.append(q_normalized, b_normalized)
        elif self._metric == "euclidean":
            q_to_pass = numpy.append(q, b)

        q_to_pass = q_to_pass.astype(numpy.float32)
        q_to_pass = numpy.ascontiguousarray(q_to_pass)

        # Get candidates from BTree
        candidate_indices, _ = self._tree_candidates.search(
            self._initial_topk, self._candidates, self._c, q_to_pass)
        
        # Convert to numpy array
        candidate_indices = numpy.array(candidate_indices, dtype=numpy.int32)
        
        # Remove any duplicates from candidate indices
        candidate_indices = numpy.unique(candidate_indices)
        
        # Use MQH to refine candidates
        # The query normalization is handled inside MQH search
        indices, _, self._num_lin_scans = self._mqh.search_with_candidates(
            q.astype(numpy.float32), n, b, self._l0, self._delta, self._flag, self._initial_candidates, candidate_indices)
        
        # Ensure no duplicates in the final result
        unique_indices = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        
        # Return only the required number of results
        return unique_indices[:n]

    def get_additional(self):
        return {
            "dist_comps": self._num_lin_scans
        }

    def __str__(self):
        return f"BT_MQH(max_leaf_size={self._max_leaf_size}, M2={self._M2}, level={self._level}, " \
               f"m_level={self._m_level}, m_num={self._m_num}, candidates={self._candidates}, " \
               f"initial_topk={self._initial_topk}, l0={self._l0}, delta={self._delta}, flag={self._flag})"

class MH_MQH(BaseANN):
    def __init__(self, metric, M_proj_vectors, m_single_hashers, l_hash_tables, 
                 M2=16, level=4, m_level=1, m_num=64):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("MH_MQH doesn't support metric %s" % metric)
        
        # MQH parameters
        self._metric = metric
        self._M2 = M2
        self._level = level
        self._m_level = m_level
        self._m_num = m_num
        self._l0 = 3  # Default parameter
        self._delta = 0.5  # Default parameter
        self._flag = 0  # Default parameter
        
        # MH parameters
        self._M_proj_vectors = M_proj_vectors
        self._m_single_hashers = m_single_hashers
        self._l_hash_tables = l_hash_tables
        self._interval_ratio = 0.9
        self._mh_index_candidates = mh.MH()
        
        # Query parameters
        self._candidates = 100  # Default
        self._initial_topk = 100  # Default

    def index(self, X):
        self._data = X.astype(numpy.float32)

        if self._metric == "angular":
            # Normalize for angular distance
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]

        # Build the MQH index for refinement
        n, d = self._data.shape
        self._mqh = mqh.MQH(d, self._M2, self._level, self._m_level, self._m_num)
        self._mqh.build_index(self._data)
        
        # For MH, add ones column after any normalization
        ones_column = numpy.ones((self._data.shape[0], 1), dtype=numpy.float32)
        self._candidate_data = numpy.hstack((self._data, ones_column))

        n, d = self._candidate_data.shape
        data_array = numpy.ascontiguousarray(self._candidate_data.ravel())
        
        # Initialize MH with parameters
        self._mh_index_candidates.preprocess(
            n, d, 
            self._M_proj_vectors, 
            self._m_single_hashers, 
            self._l_hash_tables, 
            self._interval_ratio, 
            data_array
        )

    def set_query_arguments(self, candidates=100, initial_topk=100, l0=3, delta=0.5, flag=0, initial_candidates=1):
        self._candidates = candidates
        self._initial_topk = initial_topk
        self._l0 = l0
        self._delta = delta
        self._flag = flag
        self._initial_candidates = initial_candidates

    def query(self, q, b, n):
        """
        Query using Multiprobe Hashers for candidate generation and MQH for refinement.
        
        Parameters:
        -----------
        q : numpy array
            Normal vector of the hyperplane
        b : float
            Bias term of the hyperplane equation <q, x> + b = 0
        n : int
            Number of nearest neighbors to return
            
        Returns:
        --------
        indices : list
            Indices of the n nearest neighbors to the hyperplane
        """
        # Calculate query norm
        qnorm = numpy.linalg.norm(q)

        # Prepare query vector for MH
        if self._metric == "angular":
            q_normalized = q / qnorm
            b_normalized = b / qnorm
            q_to_pass = numpy.append(q_normalized, b_normalized)
        elif self._metric == "euclidean":
            q_to_pass = numpy.append(q, b)

        q_to_pass = q_to_pass.astype(numpy.float32)
        q_to_pass = numpy.ascontiguousarray(q_to_pass)

        # Get candidates from MH
        candidate_indices, _ = self._mh_index_candidates.search(
            self._initial_topk, self._candidates, q_to_pass)
        
        # Convert to numpy array
        candidate_indices = numpy.array(candidate_indices, dtype=numpy.int32)
        
        # Remove duplicates from candidates
        candidate_indices = numpy.unique(candidate_indices)
        
        # Use MQH to refine candidates
        # The query normalization is handled inside MQH search
        indices, _, self._num_lin_scans = self._mqh.search_with_candidates(
            q.astype(numpy.float32), n, b, self._l0, self._delta, self._flag, self._initial_candidates, candidate_indices)
        
        # Ensure no duplicates in the final result
        unique_indices = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
                
        # Return only the required number of results
        return unique_indices[:n]

    def get_additional(self):
        return {
            "dist_comps": self._num_lin_scans,
        }

    def __str__(self):
        return f"MH_MQH(M_proj_vectors={self._M_proj_vectors}, m_single_hashers={self._m_single_hashers}, " \
               f"l_hash_tables={self._l_hash_tables}, M2={self._M2}, level={self._level}, " \
               f"m_level={self._m_level}, m_num={self._m_num}, candidates={self._candidates}, " \
               f"initial_topk={self._initial_topk}, l0={self._l0}, delta={self._delta}, flag={self._flag})"


class MQH(BaseANN):
    def __init__(self, metric, M2=16, level=4, m_level=1, m_num=64):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("MQH doesn't support metric %s" % metric)
        self._metric = metric
        self._M2 = M2
        self._level = level
        self._m_level = m_level
        self._m_num = m_num
        self._l0 = 3  # Default parameter
        self._delta = 0.5  # Default parameter
        self._flag = 0  # Default parameter

    def index(self, X):
        self._data = X.astype(numpy.float32)
        
        if self._metric == "angular":
            # Normalize for angular distance
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]
        
        n, d = self._data.shape
        self._mqh = mqh.MQH(d, self._M2, self._level, self._m_level, self._m_num)
        self._mqh.build_index(self._data)

    def set_query_arguments(self, l0=3, delta=0.5, flag=0, initial_candidates=1):
        self._l0 = l0
        self._delta = delta
        self._flag = flag
        self._initial_candidates = initial_candidates

    def query(self, q, b, n):
        """
        Query the index for the n nearest neighbors to the hyperplane defined by (q, b).
        
        Parameters:
        -----------
        q : numpy array
            Normal vector of the hyperplane
        b : float
            Bias term of the hyperplane equation <q, x> + b = 0
        n : int
            Number of nearest neighbors to return
            
        Returns:
        --------
        indices : numpy array
            Indices of the n nearest neighbors to the hyperplane
        """

        # Convert inputs to float32 for C++ implementation
        q = q.astype(numpy.float32)
        
        # Normalize query vector for angular distance
        if self._metric == "angular":
            qnorm = numpy.linalg.norm(q)
            if qnorm > 0:
                q = q / qnorm
                b = b / qnorm
        
        # Call the search method with the appropriate parameters
        indices, distances, self._num_lin_scans = self._mqh.search(
            q, n, b, self._l0, self._delta, self._flag, self._initial_candidates)
        
        # Remove duplicates from the results
        unique_indices = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        # Make sure we return exactly n results (or fewer if not enough unique results)
        return unique_indices[:n]
    
    def get_additional(self):
        return {
            "dist_comps": self._num_lin_scans,
        }

    def __str__(self):
        return f"MQH(M2={self._M2}, level={self._level}, m_level={self._m_level}, m_num={self._m_num}, l0={self._l0}, delta={self._delta}, flag={self._flag}, initial_candidates={self._initial_candidates})"