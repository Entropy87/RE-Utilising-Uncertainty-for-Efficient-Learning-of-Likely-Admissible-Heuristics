import random
import numpy as np
import torch
import traceback
from utils import test_15_blocksworld_improved
from neural_network import ImprovedBayesianHeuristic
from blocksworld_domain import BlocksWorld15Domain

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Starting 15-Blocksworld Test with Complete Fixed Implementation")

    
    results, trained_heuristic = test_15_blocksworld_improved()
    print("\nTest completed successfully!")
        

    print(f"- Feature dimension: {len(trained_heuristic.get_features(trained_heuristic.domain.get_goal_state()))}")
    print(f"- PDB count: {len(trained_heuristic.pdb_collection.pdbs)}")
    print(f"- Training examples: {len(trained_heuristic.memory_buffer)}")
    print(f"- Feature scales: {trained_heuristic.feature_scales}")
        
    goal_state = trained_heuristic.domain.get_goal_state()
    features = trained_heuristic.get_features(goal_state)
    print(f"- Goal state features (raw): {features}")
    print(f"- Goal state features (normalized): {trained_heuristic.normalize_features(features)}")
        
    if len(trained_heuristic.memory_buffer) > 0:
            pred, ep, al, total = trained_heuristic.predict_with_uncertainty(
                features.reshape(1, -1), num_samples=50)
            print(f"- Goal state prediction: mean={pred[0]:.3f}, epistemic={ep[0]:.3f}, aleatoric={al[0]:.3f}")
        