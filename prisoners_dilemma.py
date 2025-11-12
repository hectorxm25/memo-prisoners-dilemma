"""
One-Shot Prisoner's Dilemma Model using Memo

This model implements a one-shot prisoner's dilemma where two agents (A and B)
each make a choice to cooperate or defect. Each agent has a rationality parameter
(beta) that governs how deterministically they choose the action with higher
expected utility. Additionally, each agent has beliefs about the other agent's
rationality level.

The beta parameter (inverse temperature) in the softmax captures rationality:
- High beta (e.g., 10): Agent is highly rational, nearly always choosing optimal action
- Low beta (e.g., 0.5): Agent is irrational, choosing more randomly

Reference: Jara-Ettinger et al. (2020), "The Naive Utility Calculus as a unified,
quantitative framework for action understanding"
"""

import numpy as np
import jax.numpy as jnp
from typing import no_type_check
from memo import memo

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

# Beta parameters representing rationality levels
RATIONAL_BETA = 5.0      # High rationality: deterministic optimal choices
IRRATIONAL_BETA = 1.0    # Low rationality: more random choices

# Prisoner's Dilemma Payoff Matrix
# Standard payoff structure where T > R > P > S
T = 5  # Temptation: payoff when you defect and opponent cooperates
R = 3  # Reward: payoff when both cooperate
P = 1  # Punishment: payoff when both defect
S = 0  # Sucker: payoff when you cooperate and opponent defects

# Define action space
COOPERATE = 0
DEFECT = 1
ACTIONS = jnp.array([COOPERATE, DEFECT])

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_payoff(my_action, opponent_action):
    """
    Returns the payoff for an agent given both players' actions.
    
    Args:
        my_action: 0 for cooperate, 1 for defect
        opponent_action: 0 for cooperate, 1 for defect
    
    Returns:
        Payoff value from the perspective of 'my' agent
    """
    # Payoff matrix: rows = my action, columns = opponent action
    payoff_matrix = jnp.array([
        [R, S],  # I cooperate: get R if they cooperate, S if they defect
        [T, P]   # I defect: get T if they cooperate, P if they defect
    ])
    return payoff_matrix[my_action, opponent_action]


def softmax(utilities, beta):
    """
    Softmax function for action selection based on utilities and rationality.
    
    The probability of choosing action i is proportional to exp(beta * utility_i).
    Higher beta makes the agent more deterministic (rational), choosing the
    action with higher utility more reliably.
    
    Args:
        utilities: Array of utilities for each action
        beta: Rationality parameter (inverse temperature)
    
    Returns:
        Array of probabilities for each action
    """
    exp_values = jnp.exp(beta * utilities)
    return exp_values / jnp.sum(exp_values)


# ============================================================================
# MEMO AGENT MODELS (Simplified for Memo syntax compatibility)
# ============================================================================

# ruff: noqa
@no_type_check
@memo
def agent_choice[action: ACTIONS](prob_coop):
    """
    Simple agent model that chooses to cooperate or defect with given probability.
    
    This is a base model without recursive reasoning - used as building block.
    
    Args:
        prob_coop: Probability of cooperating (computed outside based on softmax)
    
    Returns:
        The chosen action (0=cooperate, 1=defect)
        where 0 = cooperate, 1 = defect
    """
    # Use numeric value directly since COOPERATE constant is not available in @memo context
    # action == 0 means cooperate, action == 1 means defect
    agent: chooses(action in ACTIONS, wpp=prob_coop if action == 0 else (1 - prob_coop))
    return action


# ============================================================================
# HELPER FUNCTIONS FOR COMPUTING PROBABILITIES
# ============================================================================

def compute_cooperation_prob(beta, opponent_coop_prob, debug=False):
    """
    Computes the probability that an agent will cooperate given:
    - Their own rationality (beta)
    - Their belief about the probability that the opponent will cooperate
    
    Uses softmax decision rule based on expected utilities.
    
    Args:
        beta: Agent's rationality parameter
        opponent_coop_prob: Probability agent believes opponent will cooperate
        debug: If True, print debug information
        
    Returns:
        Probability that this agent will cooperate
    """
    # Expected utility of cooperating:
    # EU(cooperate) = P(opponent cooperates) * R + P(opponent defects) * S
    eu_cooperate = opponent_coop_prob * R + (1 - opponent_coop_prob) * S
    
    # Expected utility of defecting:
    # EU(defect) = P(opponent cooperates) * T + P(opponent defects) * P
    eu_defect = opponent_coop_prob * T + (1 - opponent_coop_prob) * P
    
    if debug:
        print(f"    DEBUG: beta={beta}, opponent_coop={opponent_coop_prob:.4f}")
        print(f"    DEBUG: EU(cooperate)={eu_cooperate:.4f}, EU(defect)={eu_defect:.4f}")
        print(f"    DEBUG: EU difference (defect - cooperate)={eu_defect - eu_cooperate:.4f}")
    
    # Softmax: P(cooperate) = exp(beta * EU_coop) / (exp(beta * EU_coop) + exp(beta * EU_defect))
    # To avoid numerical overflow, use the log-sum-exp trick:
    # P(cooperate) = 1 / (1 + exp(beta * (EU_defect - EU_cooperate)))
    prob_cooperate = 1.0 / (1.0 + np.exp(beta * (eu_defect - eu_cooperate)))
    
    if debug:
        print(f"    DEBUG: P(cooperate)={prob_cooperate:.6f}")
    
    return prob_cooperate


def compute_mutual_cooperation_probs(beta_a, beta_b, belief_a_about_b, belief_b_about_a, 
                                     max_iterations=100, tolerance=1e-6, debug=False):
    """
    Computes cooperation probabilities with TRUE recursive reasoning using fixed-point iteration.
    
    In recursive reasoning:
    - A reasons: "B has rationality belief_a_about_b. If B thinks I'll cooperate with prob p_a,
                  then B will cooperate with prob p_b. Given p_b, I should cooperate with prob p_a*."
    - B reasons: "A has rationality belief_b_about_a. If A thinks I'll cooperate with prob p_b,
                  then A will cooperate with prob p_a. Given p_a, I should cooperate with prob p_b*."
    
    At equilibrium: p_a* = p_a and p_b* = p_b (fixed point / Nash equilibrium in beliefs)
    
    Args:
        beta_a: Player A's actual rationality
        beta_b: Player B's actual rationality  
        belief_a_about_b: What A believes B's rationality is
        belief_b_about_a: What B believes A's rationality is
        max_iterations: Maximum iterations for fixed-point convergence
        tolerance: Convergence tolerance
        debug: If True, show debug output
        
    Returns:
        Tuple of (prob_a_cooperates, prob_b_cooperates) at equilibrium
    """
    # Initialize with uniform prior
    prob_a = 0.5
    prob_b = 0.5
    
    if debug:
        print(f"\n  FIXED-POINT ITERATION (debug mode)")
        print(f"  Starting with prob_a={prob_a}, prob_b={prob_b}")
    
    # Iterate to find fixed point
    for iteration in range(max_iterations):
        if debug and iteration < 3:
            print(f"\n  Iteration {iteration + 1}:")
            print(f"    Current: prob_a={prob_a:.6f}, prob_b={prob_b:.6f}")
        
        # A's reasoning: Given B's believed rationality and B's belief that A cooperates with prob_a,
        # what probability will B cooperate? Then given that, what should A do?
        if debug and iteration < 3:
            print(f"    A's perspective - B will choose:")
        b_coop_from_a_perspective = compute_cooperation_prob(belief_a_about_b, prob_a, debug=(debug and iteration < 3))
        if debug and iteration < 3:
            print(f"    A's decision given B cooperates with {b_coop_from_a_perspective:.6f}:")
        new_prob_a = compute_cooperation_prob(beta_a, b_coop_from_a_perspective, debug=(debug and iteration < 3))
        
        # B's reasoning: Given A's believed rationality and A's belief that B cooperates with prob_b,
        # what probability will A cooperate? Then given that, what should B do?
        if debug and iteration < 3:
            print(f"    B's perspective - A will choose:")
        a_coop_from_b_perspective = compute_cooperation_prob(belief_b_about_a, prob_b, debug=(debug and iteration < 3))
        if debug and iteration < 3:
            print(f"    B's decision given A cooperates with {a_coop_from_b_perspective:.6f}:")
        new_prob_b = compute_cooperation_prob(beta_b, a_coop_from_b_perspective, debug=(debug and iteration < 3))
        
        # Check for convergence
        if abs(new_prob_a - prob_a) < tolerance and abs(new_prob_b - prob_b) < tolerance:
            # Converged! Return equilibrium values
            if debug:
                print(f"\n  Converged after {iteration + 1} iterations!")
                print(f"  Final: prob_a={new_prob_a:.6f}, prob_b={new_prob_b:.6f}")
            return new_prob_a, new_prob_b
        
        # Update for next iteration
        prob_a = new_prob_a
        prob_b = new_prob_b
    
    # If we didn't converge after max_iterations, return the last values
    # (In practice, this should converge quickly for reasonable beta values)
    print(f"  WARNING: Fixed-point iteration did not fully converge after {max_iterations} iterations")
    return prob_a, prob_b


# ============================================================================
# SIMULATION SCENARIOS
# ============================================================================

def run_scenario(beta_a, beta_b, belief_a_about_b, belief_b_about_a, scenario_name, debug_first=False):
    """
    Runs a single scenario and computes cooperation probabilities for both players.
    
    Args:
        beta_a: Player A's actual rationality
        beta_b: Player B's actual rationality
        belief_a_about_b: A's belief about B's rationality
        belief_b_about_a: B's belief about A's rationality
        scenario_name: Descriptive name for this scenario
        debug_first: If True, show debug output for first iteration
    
    Returns:
        Dictionary with cooperation probabilities and scenario info
    """
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")
    print(f"Player A's actual beta: {beta_a}")
    print(f"Player B's actual beta: {beta_b}")
    print(f"A's belief about B's beta: {belief_a_about_b}")
    print(f"B's belief about A's beta: {belief_b_about_a}")
    print(f"{'-'*70}")
    
    # Use TRUE recursive reasoning with fixed-point iteration
    # This finds the Nash equilibrium where each player's cooperation probability
    # is consistent with the other player's reasoning about them
    prob_a_cooperates, prob_b_cooperates = compute_mutual_cooperation_probs(
        beta_a=beta_a,
        beta_b=beta_b,
        belief_a_about_b=belief_a_about_b,
        belief_b_about_a=belief_b_about_a,
        debug=debug_first
    )
    
    # Use Memo to model the probabilistic choices (optional, for demonstration)
    # In practice, the probabilities computed above already capture the decision-making
    player_a_dist = agent_choice(prob_a_cooperates)
    player_b_dist = agent_choice(prob_b_cooperates)
    
    print(f"Player A cooperates with probability: {prob_a_cooperates:.4f}")
    print(f"Player B cooperates with probability: {prob_b_cooperates:.4f}")
    
    return {
        'scenario_name': scenario_name,
        'beta_a': beta_a,
        'beta_b': beta_b,
        'belief_a_about_b': belief_a_about_b,
        'belief_b_about_a': belief_b_about_a,
        'prob_a_cooperates': float(prob_a_cooperates),
        'prob_b_cooperates': float(prob_b_cooperates),
        'prob_a_defects': float(1 - prob_a_cooperates),
        'prob_b_defects': float(1 - prob_b_cooperates)
    }


def run_all_scenarios():
    """
    Runs all 8 scenarios:
    - 4 scenarios where both agents believe the other is rational
    - 4 scenarios where both agents believe the other is irrational
    
    Returns:
        List of dictionaries containing results for each scenario
    """
    results = []
    
    print("\n" + "="*70)
    print("PRISONER'S DILEMMA SIMULATIONS")
    print("="*70)
    
    # ========================================================================
    # SET 1: Both agents believe the other is RATIONAL
    # ========================================================================
    print("\n\nSET 1: Both agents believe opponent is RATIONAL (beta=10)")
    print("="*70)
    
    # Scenario 1: Both are actually rational, believe other is rational
    results.append(run_scenario(
        beta_a=RATIONAL_BETA,
        beta_b=RATIONAL_BETA,
        belief_a_about_b=RATIONAL_BETA,
        belief_b_about_a=RATIONAL_BETA,
        scenario_name="1. Both Rational (believe: both rational)",
        debug_first=True  # Enable debug for first scenario
    ))
    
    # Scenario 2: A rational, B irrational, both believe other is rational
    results.append(run_scenario(
        beta_a=RATIONAL_BETA,
        beta_b=IRRATIONAL_BETA,
        belief_a_about_b=RATIONAL_BETA,
        belief_b_about_a=RATIONAL_BETA,
        scenario_name="2. A Rational, B Irrational (believe: both rational)"
    ))
    
    # Scenario 3: A irrational, B rational, both believe other is rational
    results.append(run_scenario(
        beta_a=IRRATIONAL_BETA,
        beta_b=RATIONAL_BETA,
        belief_a_about_b=RATIONAL_BETA,
        belief_b_about_a=RATIONAL_BETA,
        scenario_name="3. A Irrational, B Rational (believe: both rational)"
    ))
    
    # Scenario 4: Both irrational, both believe other is rational
    results.append(run_scenario(
        beta_a=IRRATIONAL_BETA,
        beta_b=IRRATIONAL_BETA,
        belief_a_about_b=RATIONAL_BETA,
        belief_b_about_a=RATIONAL_BETA,
        scenario_name="4. Both Irrational (believe: both rational)"
    ))
    
    # ========================================================================
    # SET 2: Both agents believe the other is IRRATIONAL
    # ========================================================================
    print("\n\nSET 2: Both agents believe opponent is IRRATIONAL (beta=0.5)")
    print("="*70)
    
    # Scenario 5: Both are actually rational, believe other is irrational
    results.append(run_scenario(
        beta_a=RATIONAL_BETA,
        beta_b=RATIONAL_BETA,
        belief_a_about_b=IRRATIONAL_BETA,
        belief_b_about_a=IRRATIONAL_BETA,
        scenario_name="5. Both Rational (believe: both irrational)"
    ))
    
    # Scenario 6: A rational, B irrational, both believe other is irrational
    results.append(run_scenario(
        beta_a=RATIONAL_BETA,
        beta_b=IRRATIONAL_BETA,
        belief_a_about_b=IRRATIONAL_BETA,
        belief_b_about_a=IRRATIONAL_BETA,
        scenario_name="6. A Rational, B Irrational (believe: both irrational)"
    ))
    
    # Scenario 7: A irrational, B rational, both believe other is irrational
    results.append(run_scenario(
        beta_a=IRRATIONAL_BETA,
        beta_b=RATIONAL_BETA,
        belief_a_about_b=IRRATIONAL_BETA,
        belief_b_about_a=IRRATIONAL_BETA,
        scenario_name="7. A Irrational, B Rational (believe: both irrational)"
    ))
    
    # Scenario 8: Both irrational, both believe other is irrational
    results.append(run_scenario(
        beta_a=IRRATIONAL_BETA,
        beta_b=IRRATIONAL_BETA,
        belief_a_about_b=IRRATIONAL_BETA,
        belief_b_about_a=IRRATIONAL_BETA,
        scenario_name="8. Both Irrational (believe: both irrational)"
    ))
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ONE-SHOT PRISONER'S DILEMMA WITH RECURSIVE REASONING")
    print("="*70)
    print(f"\nPayoff Matrix:")
    print(f"  Both Cooperate: ({R}, {R})")
    print(f"  A Defects, B Cooperates: ({T}, {S})")
    print(f"  A Cooperates, B Defects: ({S}, {T})")
    print(f"  Both Defect: ({P}, {P})")
    print(f"\nRationality Parameters:")
    print(f"  RATIONAL_BETA = {RATIONAL_BETA}")
    print(f"  IRRATIONAL_BETA = {IRRATIONAL_BETA}")
    
    # Run all scenarios
    all_results = run_all_scenarios()
    
    # Save results for plotting
    import json
    with open('simulation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Results saved to: simulation_results.json")
    print(f"Run 'python plot_results.py' to generate plots.")

