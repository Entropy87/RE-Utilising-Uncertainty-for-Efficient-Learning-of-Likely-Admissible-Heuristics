from blocksworld_domain import BlocksWorld15Domain
import numpy as np
import time

class ImprovedIDAStar:
    """IDA* search implementation with better error handling."""
    
    def __init__(self, domain, heuristic_function):
        self.domain = domain
        self.heuristic = heuristic_function
        self.expanded = 0
        self.generated = 0
        
    def search(self, start_state, max_time=300):
        """Perform IDA* search from start state."""
        self.expanded = 0
        self.generated = 0
        start_time = time.time()
        bound = self.heuristic(start_state)
        path = [start_state]
        
        while True:
            if max_time is not None and time.time() - start_time > max_time:
                return []
            t = self._dfs(path, 0, bound, start_time, max_time)
            if t == True:
                return path
            if t == float('inf') or t is None:
                return []
            bound = t
            
    def _dfs(self, path, g, bound, start_time, max_time):
        """Recursive DFS with iterative deepening."""
        if max_time is not None and time.time() - start_time > max_time:
            return None
        node = path[-1]
        try:
            f = g + self.heuristic(node)
        except Exception as e:
            print(f"Heuristic evaluation failed: {e}")
            return float('inf')
        
        if f > bound:
            return f
        if self.domain.is_goal(node):
            return True
        self.expanded += 1
        min_cost = float('inf')
        
        try:
            operators = self.domain.get_operators(node)
        except Exception as e:
            print(f"Operator generation failed: {e}")
            return float('inf')
            
        for op in operators:
            try:
                child = self.domain.apply_operator(node, op)
            except Exception as e:
                print(f"Operator application failed: {e}")
                continue
            cycle_check_depth = min(10, len(path))
            if any(np.array_equal(child, path[-(i+1)]) for i in range(cycle_check_depth)):
                continue
            self.generated += 1
            path.append(child)
            t = self._dfs(path, g + 1, bound, start_time, max_time)
            if t == True:
                return True
            if t is None:
                path.pop()
                return None
            if t < min_cost:
                min_cost = t
            path.pop()
        return min_cost