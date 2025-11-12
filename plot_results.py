"""
Plot Results for Prisoner's Dilemma Simulations

This script reads the simulation results from prisoners_dilemma.py and generates
8 plots (one for each scenario) showing both Player A's and Player B's
cooperation probabilities.

Each plot displays:
- Player A's probability of cooperating (blue bar)
- Player B's probability of cooperating (orange bar)
- Player A's probability of defecting (light blue bar)
- Player B's probability of defecting (light orange bar)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# LOAD SIMULATION RESULTS
# ============================================================================

def load_results():
    """
    Loads simulation results from JSON file.
    
    Returns:
        List of dictionaries containing scenario results
    """
    results_file = Path('simulation_results.json')
    
    if not results_file.exists():
        print("Error: simulation_results.json not found!")
        print("Please run 'python prisoners_dilemma.py' first to generate results.")
        exit(1)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_scenario_plot(result, scenario_idx, total_scenarios):
    """
    Creates a bar plot for a single scenario showing cooperation/defection
    probabilities for both players.
    
    Args:
        result: Dictionary containing scenario results
        scenario_idx: Index of this scenario (0-based)
        total_scenarios: Total number of scenarios for subplot layout
    """
    # Extract data
    scenario_name = result['scenario_name']
    prob_a_coop = result['prob_a_cooperates']
    prob_b_coop = result['prob_b_cooperates']
    prob_a_defect = result['prob_a_defects']
    prob_b_defect = result['prob_b_defects']
    
    # Create figure for this scenario
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    players = ['Player A', 'Player B']
    x = np.arange(len(players))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, [prob_a_coop, prob_b_coop], width, 
                   label='Cooperate', color=['#2E86AB', '#A23B72'], alpha=0.8)
    bars2 = ax.bar(x + width/2, [prob_a_defect, prob_b_defect], width,
                   label='Defect', color=['#87CEEB', '#DDA0DD'], alpha=0.8)
    
    # Customize plot
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'{scenario_name}\n' + 
                 f'(A: β={result["beta_a"]}, believes B has β={result["belief_a_about_b"]}; ' +
                 f'B: β={result["beta_b"]}, believes A has β={result["belief_b_about_a"]})',
                 fontsize=11, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(players, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add a subtle background color to distinguish the two sets
    if scenario_idx < 4:
        # First 4 scenarios: believe opponent is rational
        fig.patch.set_facecolor('#FFF8DC')  # Cornsilk
        ax.set_facecolor('#FFFAF0')  # Floral white
    else:
        # Last 4 scenarios: believe opponent is irrational
        fig.patch.set_facecolor('#F0F8FF')  # Alice blue
        ax.set_facecolor('#F8F8FF')  # Ghost white
    
    plt.tight_layout()
    
    # Save individual plot
    filename = f'scenario_{scenario_idx + 1}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    
    return fig


def create_combined_comparison_plot(results):
    """
    Creates a combined plot showing all scenarios in a 2x4 grid for easy comparison.
    
    Args:
        results: List of all scenario results
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('One-Shot Prisoner\'s Dilemma: Cooperation Probabilities Across All Scenarios',
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, (result, ax) in enumerate(zip(results, axes.flat)):
        # Extract data
        prob_a_coop = result['prob_a_cooperates']
        prob_b_coop = result['prob_b_cooperates']
        
        # Set up bars
        players = ['A', 'B']
        x = np.arange(len(players))
        width = 0.7
        
        # Create bars for cooperation only (simplified for comparison view)
        bars = ax.bar(x, [prob_a_coop, prob_b_coop], width,
                     color=['#2E86AB', '#A23B72'], alpha=0.8)
        
        # Customize subplot
        ax.set_ylabel('P(Cooperate)', fontsize=9)
        ax.set_title(f'{idx + 1}. β_A={result["beta_a"]}, β_B={result["beta_b"]}\n' +
                    f'Beliefs: {result["belief_a_about_b"]}, {result["belief_b_about_a"]}',
                    fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(players, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=7)
        
        # Color background based on belief type
        if idx < 4:
            ax.set_facecolor('#FFFAF0')  # Believe rational
        else:
            ax.set_facecolor('#F8F8FF')  # Believe irrational
    
    plt.tight_layout()
    
    # Save combined plot
    filename = 'all_scenarios_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    
    return fig


def create_belief_comparison_plot(results):
    """
    Creates a comparison plot showing how beliefs affect cooperation.
    Compares scenarios with same actual betas but different beliefs.
    
    Args:
        results: List of all scenario results
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Effect of Beliefs on Cooperation: Same Reality, Different Beliefs',
                 fontsize=14, fontweight='bold')
    
    # Define the 4 reality conditions
    conditions = [
        (10.0, 10.0, "Both Actually Rational (β=10, β=10)"),
        (10.0, 0.5, "A Rational, B Irrational (β=10, β=0.5)"),
        (0.5, 10.0, "A Irrational, B Rational (β=0.5, β=10)"),
        (0.5, 0.5, "Both Actually Irrational (β=0.5, β=0.5)")
    ]
    
    for idx, (beta_a, beta_b, title) in enumerate(conditions):
        ax = axes[idx // 2, idx % 2]
        
        # Find scenarios matching this reality
        matching_scenarios = [r for r in results 
                            if r['beta_a'] == beta_a and r['beta_b'] == beta_b]
        
        if len(matching_scenarios) != 2:
            print(f"Warning: Expected 2 scenarios for {title}, found {len(matching_scenarios)}")
            continue
        
        # Sort by belief (rational belief first)
        matching_scenarios.sort(key=lambda x: x['belief_a_about_b'], reverse=True)
        
        # Extract data
        belief_labels = [f"Believe\nRational\n(β={s['belief_a_about_b']})" 
                        if s['belief_a_about_b'] == 10.0 
                        else f"Believe\nIrrational\n(β={s['belief_a_about_b']})"
                        for s in matching_scenarios]
        
        a_probs = [s['prob_a_cooperates'] for s in matching_scenarios]
        b_probs = [s['prob_b_cooperates'] for s in matching_scenarios]
        
        # Set up bars
        x = np.arange(len(belief_labels))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, a_probs, width, label='Player A', 
                      color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, b_probs, width, label='Player B',
                      color='#A23B72', alpha=0.8)
        
        # Customize
        ax.set_ylabel('P(Cooperate)', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(belief_labels, fontsize=8)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    # Save comparison plot
    filename = 'belief_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to load results and generate all plots.
    """
    print("\n" + "="*70)
    print("GENERATING PLOTS FOR PRISONER'S DILEMMA SIMULATIONS")
    print("="*70)
    
    # Load results
    print("\nLoading simulation results...")
    results = load_results()
    print(f"Loaded {len(results)} scenarios")
    
    # Create individual plots for each scenario
    print("\n" + "-"*70)
    print("Creating individual scenario plots...")
    print("-"*70)
    for idx, result in enumerate(results):
        create_scenario_plot(result, idx, len(results))
    
    # Create combined comparison plot
    print("\n" + "-"*70)
    print("Creating combined comparison plot...")
    print("-"*70)
    create_combined_comparison_plot(results)
    
    # Create belief comparison plot
    print("\n" + "-"*70)
    print("Creating belief comparison plot...")
    print("-"*70)
    create_belief_comparison_plot(results)
    
    print("\n" + "="*70)
    print("PLOTTING COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - scenario_1.png through scenario_8.png (individual scenarios)")
    print("  - all_scenarios_combined.png (all scenarios in one view)")
    print("  - belief_comparison.png (effect of beliefs on cooperation)")
    print("\n")


if __name__ == "__main__":
    main()

