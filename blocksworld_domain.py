import numpy as np

class BlocksWorld15Domain:
    """15-blocksworld implementation with consistent indexing."""
    
    def __init__(self):
        self.num_blocks = 15
        
    def get_goal_state(self):
        """Returns the goal state where:
        - state[i] represents what block (i+1) is on
        - 0 means on table
        - Goal: block 1 on table, block 2 on block 1, etc.
        """
        return np.arange(self.num_blocks, dtype=np.int8)  # [0,1,2,3,...,14]
        
    def is_goal(self, state):
        """Check if state is the goal state."""
        return np.array_equal(state, self.get_goal_state())
        
    def state_to_stacks(self, state):
        """Convert state array to list of stacks.
        Returns stacks with 1-indexed block numbers for clarity.
        """
        stacks = []
        
        # Find all blocks on table (state[i] == 0 means block i+1 is on table)
        on_table = []
        for i in range(self.num_blocks):
            if state[i] == 0:
                on_table.append(i + 1)  # Convert to 1-indexed block number
        
        # Build stacks from bottom up
        for bottom_block in on_table:
            stack = [bottom_block]
            current_block = bottom_block
            
            # Follow the chain upward
            while True:
                found_above = False
                for i in range(self.num_blocks):
                    # If state[i] == current_block, then block (i+1) is on current_block
                    if state[i] == current_block:
                        next_block = i + 1
                        stack.append(next_block)
                        current_block = next_block
                        found_above = True
                        break
                        
                if not found_above:
                    break
            
            stacks.append(stack)
        
        return stacks
    
    def get_operators(self, state):
        """Get all legal moves (block, destination, original_below).
        All values use 1-indexed block numbers, 0 = table.
        """
        stacks = self.state_to_stacks(state)
        operators = []
        
        # Find moveable blocks (top of each stack)
        moveable_blocks = [stack[-1] for stack in stacks if stack]
        
        for block in moveable_blocks:
            original_below = state[block - 1]  # What this block is currently on
            
            # Can move to top of any other stack (not the same stack)
            for other_stack in stacks:
                if other_stack and other_stack[-1] != block:  # Not the same stack
                    target_block = other_stack[-1]
                    operators.append((block, target_block, original_below))
            
            # Can move to table (if not already on table)
            if original_below != 0:
                operators.append((block, 0, original_below))
        
        return operators
        
    def apply_operator(self, state, operator):
        """Apply a move (block, destination, original_below).
        All values use 1-indexed block numbers, 0 = table.
        """
        if not (1 <= operator[0] <= self.num_blocks):
            raise ValueError(f"Invalid block number: {operator[0]}")
        
        if operator[1] != 0 and not (1 <= operator[1] <= self.num_blocks):
            raise ValueError(f"Invalid destination: {operator[1]}")
            
        new_state = state.copy()
        block, destination, _ = operator
        new_state[block - 1] = destination  # Convert to 0-indexed array access
        return new_state
        
    def hamming_distance(self, state):
        """Calculate number of blocks not in goal position."""
        goal = self.get_goal_state()
        return sum(1 for i in range(self.num_blocks) if state[i] != goal[i])