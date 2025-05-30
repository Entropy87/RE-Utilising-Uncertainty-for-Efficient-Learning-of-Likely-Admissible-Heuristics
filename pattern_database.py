from collections import deque
from blocksworld_domain import BlocksWorld15Domain
import numpy as np

class PatternDatabase:
    """Pattern database for blocksworld with backward search."""
    
    def __init__(self, pattern_blocks, domain):
        self.pattern_blocks = set(pattern_blocks)
        self.domain = domain
        self.pdb = {}
        self.max_value = 0
        self._build_pdb()
    
    def _abstract_state(self, state):
        """Abstract state to only include pattern blocks."""
        abstract = {}
        for block in self.pattern_blocks:
            if 1 <= block <= self.domain.num_blocks:
                below = state[block - 1]
                if below == 0:
                    abstract[block] = 0
                elif below in self.pattern_blocks:
                    abstract[block] = below
                else:
                    abstract[block] = -1
        return tuple(sorted(abstract.items()))
    
    def _build_pdb(self):
        """Build pattern database using backward search from goal."""
        goal_state = self.domain.get_goal_state()
        abstract_goal = self._abstract_state(goal_state)
        self.pdb = {abstract_goal: 0}
        queue = deque([(abstract_goal, 0)])
        max_states = 50000
        states_explored = 0
        
        print(f"Building PDB for pattern {sorted(self.pattern_blocks)}...")
        
        while queue and states_explored < max_states:
            current_abstract, cost = queue.popleft()
            states_explored += 1
            
            if states_explored % 10000 == 0:
                print(f"  Explored {states_explored} states, queue size: {len(queue)}")
            
            predecessors = self._generate_predecessors(current_abstract, cost)
            
            for pred_abstract in predecessors:
                if pred_abstract not in self.pdb:
                    new_cost = cost + 1
                    self.pdb[pred_abstract] = new_cost
                    self.max_value = max(self.max_value, new_cost)
                    queue.append((pred_abstract, new_cost))
        
        print(f"  PDB built with {len(self.pdb)} states, max value: {self.max_value}")
    
    def _generate_predecessors(self, abstract_state, current_cost):
        """Generate predecessor abstract states."""
        predecessors = set()
        state_dict = dict(abstract_state)
        
        for block in self.pattern_blocks:
            if block not in state_dict:
                continue
                
            current_below = state_dict[block]
            possible_destinations = [0]
            possible_destinations.extend(self.pattern_blocks)
            possible_destinations.append(-1)
            
            for new_below in possible_destinations:
                if new_below == current_below:
                    continue
                new_state_dict = state_dict.copy()
                new_state_dict[block] = new_below
                if self._is_valid_abstract_state(new_state_dict):
                    pred_abstract = tuple(sorted(new_state_dict.items()))
                    predecessors.add(pred_abstract)
        
        return predecessors
    
    def _is_valid_abstract_state(self, state_dict):
        """Check if abstract state is valid."""
        visited = set()
        for block in state_dict:
            if block in visited:
                continue
            current = block
            chain = set()
            while current in state_dict and current != 0 and current != -1:
                if current in chain:
                    return False
                chain.add(current)
                current = state_dict[current]
                if current not in self.pattern_blocks and current != 0 and current != -1:
                    break
            visited.update(chain)
        return True
    
    def get_heuristic(self, state):
        """Get heuristic value for state."""
        abstract = self._abstract_state(state)
        if abstract in self.pdb:
            return self.pdb[abstract]
        h = 0
        goal = self.domain.get_goal_state()
        for block in self.pattern_blocks:
            if 1 <= block <= self.domain.num_blocks:
                if state[block - 1] != goal[block - 1]:
                    h += 1
        return h

class PDBCollection:
    """Collection of PDBs matching paper's implementation."""
    
    def __init__(self, domain):
        self.patterns = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [12, 13, 14, 15],
            [3, 4, 5, 6],
            [7, 8, 9, 10],
            [11, 12, 13, 14],
            [1, 2, 14, 15],
            [2, 3, 4, 5],
            [6, 7, 8, 9],
            [10, 11, 12, 13],
            [3, 6, 9],
            [1, 13, 14, 15]
        ]
        self.domain = domain
        print(f"Creating {len(self.patterns)} pattern databases...")
        
        self.pdbs = []
        for i, pattern in enumerate(self.patterns):
            print(f"Building PDB {i+1}/{len(self.patterns)} for pattern {pattern}")
            try:
                pdb = PatternDatabase(pattern, domain)
                self.pdbs.append(pdb)
                print(f"  Success: {len(pdb.pdb)} states, max value {pdb.max_value}")
            except Exception as e:
                print(f"  Failed to build PDB for pattern {pattern}: {e}")
                dummy_pdb = type('DummyPDB', (), {'get_heuristic': lambda self, state: 0, 'max_value': 0})()
                self.pdbs.append(dummy_pdb)
        
        self.max_values = []
        for pdb in self.pdbs:
            if hasattr(pdb, 'max_value'):
                self.max_values.append(pdb.max_value)
            else:
                self.max_values.append(1)
        
        self.max_values.append(domain.num_blocks)
        self.max_values.append(domain.num_blocks)
        
        print(f"PDB Collection created with max values: {self.max_values}")
    
    def get_heuristics(self, state):
        """Get all PDB heuristic values."""
        heuristics = []
        for pdb in self.pdbs:
            try:
                h = pdb.get_heuristic(state)
                heuristics.append(h)
            except Exception as e:
                print(f"PDB evaluation failed: {e}")
                heuristics.append(0)
        return heuristics