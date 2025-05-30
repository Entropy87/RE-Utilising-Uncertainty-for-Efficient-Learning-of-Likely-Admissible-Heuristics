import numpy as np
import random
import time
import logging
from blocksworld_domain import BlocksWorld15Domain
from neural_network import ImprovedBayesianHeuristic
from ida_star import ImprovedIDAStar
from pattern_database import PDBCollection

def setup_logging():
    """Setup logging for better debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('blocksworld_training.log'),
            logging.StreamHandler()
        ]
    )

def generate_task_improved(heuristic, domain, epsilon=1.0, max_steps=50):
    """Generate task using epistemic uncertainty."""
    state = domain.get_goal_state()
    visited = set([tuple(state)])
    
    for step in range(max_steps):
        operators = domain.get_operators(state)
        if not operators:
            break
        next_states = []
        for op in operators:
            new_state = domain.apply_operator(state, op)
            state_tuple = tuple(new_state)
            if state_tuple not in visited:
                next_states.append(new_state)
                visited.add(state_tuple)
        
        if not next_states:
            next_states = [domain.apply_operator(state, op) for op in operators]
        
        if not next_states:
            break
        
        if len(heuristic.memory_buffer) > 50:
            try:
                features = np.array([heuristic.get_features(s) for s in next_states])
                _, epistemic, _, _ = heuristic.predict_with_uncertainty(features, num_samples=100)
                max_uncertainty = np.max(epistemic)
                if max_uncertainty >= epsilon:
                    idx = np.argmax(epistemic)
                    print(f"    Task generated after {step+1} steps with uncertainty {max_uncertainty:.4f}")
                    return next_states[idx]
                exp_uncertainties = np.exp(epistemic - np.max(epistemic))
                if np.sum(exp_uncertainties) > 0:
                    probs = exp_uncertainties / np.sum(exp_uncertainties)
                    idx = np.random.choice(len(next_states), p=probs)
                else:
                    idx = np.random.choice(len(next_states))
            except Exception as e:
                print(f"    Uncertainty computation failed: {e}, using random selection")
                idx = np.random.choice(len(next_states))
        else:
            idx = np.random.choice(len(next_states))
        
        state = next_states[idx]
    
    print(f"    Task generation reached max steps ({max_steps}).")
    return state

def generate_test_tasks(domain, num_tasks=50, min_steps=15, max_steps=40):
    """Generate test tasks with appropriate difficulty and better validation."""
    test_states = []
    optimal_lengths = []
    pdb_collection = PDBCollection(domain)
    
    attempts = 0
    max_attempts = num_tasks
    
    while len(test_states) < num_tasks and attempts < max_attempts:
        attempts += 1
        state = domain.get_goal_state()
        visited = set([tuple(state)])
        target_steps = random.randint(min_steps, max_steps)
        actual_steps = 0
        
        for _ in range(target_steps):
            ops = domain.get_operators(state)
            if not ops:
                break
            valid_ops = []
            for op in ops:
                next_state = domain.apply_operator(state, op)
                if tuple(next_state) not in visited:
                    valid_ops.append(op)
            if valid_ops:
                op = random.choice(valid_ops)
            else:
                op = random.choice(ops)
            state = domain.apply_operator(state, op)
            visited.add(tuple(state))
            actual_steps += 1
        
        if actual_steps >= min_steps:
            pdb_values = pdb_collection.get_heuristics(state)
            estimated_optimal = max(pdb_values) if pdb_values else 1
            if estimated_optimal > 0:
                test_states.append(state)
                optimal_lengths.append(estimated_optimal)
                print(f"Generated test problem {len(test_states)}: "
                      f"{actual_steps} steps, estimated optimal {estimated_optimal}")
    
    if len(test_states) < num_tasks:
        print(f"Warning: Only generated {len(test_states)} out of {num_tasks} requested tasks")
    
    return test_states, optimal_lengths

def learn_heuristic_improved(domain, num_iterations=75, num_tasks_per_iter=10, 
                           epsilon=1.0, alpha_initial=0.99, max_time=300):
    """Learn a heuristic using uncertainty-based task generation."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with {num_iterations} iterations")
    heuristic = ImprovedBayesianHeuristic(domain, input_dim=14, hidden_dim=8)
    alpha = alpha_initial
    min_tasks_to_solve = int(0.6 * num_tasks_per_iter)
    all_costs = []
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")
        print(f"Current alpha: {alpha:.4f}")
        plans = []
        num_solved = 0
        iteration_start_time = time.time()
        
        for task_idx in range(num_tasks_per_iter):
            print(f"  Generating task {task_idx+1}/{num_tasks_per_iter}...")
            if len(heuristic.memory_buffer) > 20:
                task_state = generate_task_improved(heuristic, domain, epsilon, max_steps=50)
            else:
                task_state = domain.get_goal_state()
                for _ in range(random.randint(15, 25)):
                    ops = domain.get_operators(task_state)
                    if ops:
                        op = random.choice(ops)
                        task_state = domain.apply_operator(task_state, op)
            
            def h_func(state):
                try:
                    features = heuristic.get_features(state).reshape(1, -1)
                    if len(heuristic.memory_buffer) > 50:
                        if alpha < 1.0:
                            h_val, _, _, _ = heuristic.predict_with_uncertainty(
                                features, alpha=alpha, num_samples=100)
                            return max(0, int(h_val[0]))
                        else:
                            h_val, _, _, _ = heuristic.predict_with_uncertainty(
                                features, num_samples=100)
                            return max(0, int(h_val[0]))
                    else:
                        return max(heuristic.pdb_collection.get_heuristics(state))
                except Exception as e:
                    print(f"    Heuristic prediction failed: {e}")
                    return max(heuristic.pdb_collection.get_heuristics(state))
            
            print(f"  Solving task {task_idx+1}...")
            planner = ImprovedIDAStar(domain, h_func)
            try:
                plan = planner.search(task_state, max_time=max_time)
                if plan and len(plan) > 1:
                    num_solved += 1
                    plans.append(plan)
                    cost = len(plan) - 1
                    all_costs.append(cost)
                    print(f"    Task {task_idx+1} solved! Plan length: {cost}")
                    print(f"    Expanded: {planner.expanded:,}, Generated: {planner.generated:,}")
                else:
                    print(f"    Task {task_idx+1} failed: no valid plan found")
            except Exception as e:
                print(f"    Task {task_idx+1} failed with error: {e}")
        
        print(f"Solved {num_solved}/{num_tasks_per_iter} tasks in iteration {iteration+1}")
        if num_solved < min_tasks_to_solve:
            alpha = max(0.5, alpha - 0.05)
            print(f"Too few tasks solved, decreasing alpha to {alpha:.4f}")
        elif num_solved == num_tasks_per_iter and alpha < alpha_initial:
            alpha = min(alpha_initial, alpha + 0.02)
            print(f"All tasks solved, increasing alpha to {alpha:.4f}")
        
        if plans:
            features_list = []
            costs_list = []
            for plan in plans:
                for i in range(len(plan)):
                    state = plan[i]
                    cost_to_goal = len(plan) - 1 - i
                    features = heuristic.get_features(state)
                    features_list.append(features)
                    costs_list.append(cost_to_goal)
            if features_list:
                heuristic.add_to_memory_buffer(features_list, costs_list)
                print(f"  Added {len(features_list)} training examples")
        
        if len(heuristic.memory_buffer) >= 100:
            print("  Training heuristic networks...")
            try:
                X = np.array([x for x, _ in heuristic.memory_buffer])
                y = np.array([y for _, y in heuristic.memory_buffer])
                if len(X) > 25000:
                    indices = np.random.choice(len(X), 25000, replace=False)
                    X = X[indices]
                    y = y[indices]
                heuristic.train_networks(X, y, epochs=800)
                print(f"  Training completed on {len(X)} examples")
            except Exception as e:
                print(f"  Training failed: {e}")
        
        iteration_time = time.time() - iteration_start_time
        print(f"Iteration {iteration+1} completed in {iteration_time:.1f} seconds")
    
    print(f"\nTraining completed!")
    print(f"Final buffer size: {len(heuristic.memory_buffer)}")
    if all_costs:
        print(f"Average plan cost: {np.mean(all_costs):.2f}")
    
    return heuristic

def test_15_blocksworld_improved():
    """Test function to reproduce Table 11 results."""
    print("Improved 15-Blocksworld Implementation Test")
    print("=" * 80)
    setup_logging()
    domain = BlocksWorld15Domain()
    
    print("Learning heuristic...")
    start_time = time.time()
    heuristic = learn_heuristic_improved(
        domain,
        num_iterations=5,
        num_tasks_per_iter=5,
        epsilon=0.5,
        alpha_initial=0.95,
        max_time=60
    )
    training_time = time.time() - start_time
    print(f"\nHeuristic learning completed in {training_time:.1f} seconds")
    
    print("\nGenerating test problems...")
    test_states, optimal_lengths = generate_test_tasks(
        domain, num_tasks=10, min_steps=10, max_steps=20)
    
    alpha_values = [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, None]
    results = {}
    
    for alpha in alpha_values:
        print(f"\nTesting with alpha = {alpha}")
        times = []
        generated_nodes = []
        expanded_nodes = []
        suboptimalities = []
        solved_count = 0
        
        for i, state in enumerate(test_states):
            print(f"  Test problem {i+1}/{len(test_states)}...")
            def h_func(state):
                try:
                    features = heuristic.get_features(state).reshape(1, -1)
                    baseline_h = max(heuristic.pdb_collection.get_heuristics(state))
                    if len(heuristic.memory_buffer) > 50:
                        if alpha is not None:
                            h_val, _, _, _ = heuristic.predict_with_uncertainty(
                                features, alpha=alpha, num_samples=100)
                            predicted_h = max(0, int(h_val[0]))
                        else:
                            h_val, _, _, _ = heuristic.predict_with_uncertainty(
                                features, num_samples=100)
                            predicted_h = max(0, int(h_val[0]))
                        return max(predicted_h, baseline_h)
                    else:
                        return baseline_h
                except Exception as e:
                    print(f"    Prediction error: {e}, using baseline")
                    return max(heuristic.pdb_collection.get_heuristics(state))
            
            planner = ImprovedIDAStar(domain, h_func)
            test_start_time = time.time()
            try:
                plan = planner.search(state, max_time=60)
                solve_time = time.time() - test_start_time
                if plan and len(plan) > 1:
                    solved_count += 1
                    times.append(solve_time)
                    generated_nodes.append(planner.generated)
                    expanded_nodes.append(planner.expanded)
                    found_length = len(plan) - 1
                    estimated_optimal = optimal_lengths[i]
                    if estimated_optimal > 0:
                        subopt = max(0, (found_length / estimated_optimal) - 1)
                    else:
                        subopt = 0
                    suboptimalities.append(subopt)
                    print(f"    Solved: {found_length} moves, time: {solve_time:.1f}s")
                    print(f"    Expanded: {planner.expanded:,}, Generated: {planner.generated:,}")
                else:
                    print(f"    Failed to solve within time limit")
            except Exception as e:
                print(f"    Search failed with error: {e}")
        
        if times:
            results[alpha] = {
                'avg_time': np.mean(times),
                'avg_generated': np.mean(generated_nodes),
                'avg_expanded': np.mean(expanded_nodes),
                'avg_subopt': np.mean(suboptimalities) if suboptimalities else 0.0,
                'solved_pct': (solved_count / len(test_states)) * 100,
                'solved_count': solved_count
            }
        else:
            results[alpha] = {
                'avg_time': 0.0,
                'avg_generated': 0,
                'avg_expanded': 0,
                'avg_subopt': 0.0,
                'solved_pct': 0,
                'solved_count': 0
            }
    
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"{'Alpha':<8} {'Solved':<8} {'Time':<10} {'Expanded':<12} {'Generated':<12} {'Subopt%':<10}")
    print("-"*100)
    
    for alpha in alpha_values:
        if alpha in results:
            r = results[alpha]
            alpha_str = 'Mean' if alpha is None else f'{alpha:.2f}'
            print(f"{alpha_str:<8} {r['solved_count']:<8} {r['avg_time']:<10.1f} "
                  f"{r['avg_expanded']:<12,.0f} {r['avg_generated']:<12,.0f} "
                  f"{r['avg_subopt']*100:<9.1f}%")
    
    print(f"\nTotal training time: {training_time:.1f} seconds")
    print(f"Memory buffer final size: {len(heuristic.memory_buffer)}")
    
    return results, heuristic