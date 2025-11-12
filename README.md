# One-Shot Prisoner's Dilemma with Recursive Reasoning in Memo

This project models a one-shot prisoner's dilemma game where two agents (Player A and Player B) make simultaneous choices to either cooperate or defect. The key innovation is that each agent has both:
1. **Their own rationality level** (β parameter)
2. **Beliefs about the opponent's rationality level**

This implements recursive reasoning: "I think you have rationality X, so you'll do Y, therefore I should do Z."

## Background

### The Prisoner's Dilemma

The prisoner's dilemma is a fundamental game in game theory where two players must simultaneously choose to cooperate or defect. The payoff structure creates a tension between individual and collective rationality:

| | Opponent Cooperates | Opponent Defects |
|---|---|---|
| **You Cooperate** | R=3, R=3 | S=0, T=5 |
| **You Defect** | T=5, S=0 | P=1, P=1 |

Where: T (Temptation) > R (Reward) > P (Punishment) > S (Sucker)

The dilemma: Defecting is always better individually, but both players are better off if they both cooperate.

### The Beta Parameter and Rationality

The beta (β) parameter, also known as the inverse temperature parameter in softmax decision-making, captures an agent's rationality level:

- **High β (e.g., 10)**: The agent is highly rational, nearly always choosing the action with higher expected utility
- **Low β (e.g., 0.5)**: The agent is irrational, choosing more randomly among available actions

The probability of choosing an action is given by the softmax function:

```
P(action) = exp(β × utility(action)) / Σ exp(β × utility(all actions))
```

**Reference**: Jara-Ettinger, J., Gweon, H., Schulz, L. E., & Tenenbaum, J. B. (2020). "The Naive Utility Calculus as a unified, quantitative framework for action understanding." *Cognitive Psychology, 123*, 101334.

This paper demonstrates how the β parameter captures observers' intuitions about rationality in action understanding.

## Project Structure

This project consists of three files:

1. **`prisoners_dilemma.py`**: Main Memo model implementing agents, beliefs, and simulations
2. **`plot_results.py`**: Python script to generate visualizations
3. **`README.md`**: This documentation file

## Installation

### Prerequisites

1. **Python 3.12 or higher**
   ```bash
   python --version  # Check your version
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install JAX** (required by Memo)
   - Without GPU:
     ```bash
     pip install jax
     ```
   - With GPU: See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)

5. **Install Memo**
   ```bash
   pip install memo-lang
   ```

6. **Install plotting dependencies**
   ```bash
   pip install matplotlib numpy
   ```

## Usage

### Step 1: Run the Simulation

Execute the main Memo model to simulate all 8 scenarios:

```bash
python prisoners_dilemma.py
```

This will:
- Define the prisoner's dilemma game structure
- Create agent models with recursive reasoning
- Simulate all 8 scenarios (described below)
- Print detailed results to console
- Save results to `simulation_results.json`

**Expected runtime**: 1-5 minutes depending on your hardware

### Step 2: Generate Plots

After running the simulation, generate visualizations:

```bash
python plot_results.py
```

This will create:
- `scenario_1.png` through `scenario_8.png`: Individual plots for each scenario
- `all_scenarios_combined.png`: All scenarios in a 2×4 grid for comparison
- `belief_comparison.png`: Shows how beliefs affect cooperation within each reality condition

## The Eight Scenarios

The simulation explores 8 scenarios organized into two sets:

### Set 1: Both agents believe opponent is RATIONAL (β=10)

1. **Both Actually Rational**: Both have β=10
   - Reality matches beliefs
   
2. **A Rational, B Irrational**: A has β=10, B has β=0.5
   - A is correct about themselves, wrong about B
   - B is wrong about themselves, correct about A
   
3. **A Irrational, B Rational**: A has β=0.5, B has β=10
   - Opposite of scenario 2
   
4. **Both Actually Irrational**: Both have β=0.5
   - Reality contradicts beliefs

### Set 2: Both agents believe opponent is IRRATIONAL (β=0.5)

5. **Both Actually Rational**: Both have β=10
   - Reality contradicts beliefs
   
6. **A Rational, B Irrational**: A has β=10, B has β=0.5
   - A is correct about themselves, wrong about B
   - B is wrong about themselves, correct about A
   
7. **A Irrational, B Rational**: A has β=0.5, B has β=10
   - Opposite of scenario 6
   
8. **Both Actually Irrational**: Both have β=0.5
   - Reality matches beliefs

## Modifying Parameters

You can easily modify the rationality parameters in `prisoners_dilemma.py`:

```python
# At the top of prisoners_dilemma.py
RATIONAL_BETA = 10.0      # Change this value
IRRATIONAL_BETA = 0.5     # Change this value
```

You can also modify the payoff matrix:

```python
# In prisoners_dilemma.py
T = 5  # Temptation
R = 3  # Reward
P = 1  # Punishment
S = 0  # Sucker
```

After modifying parameters, re-run both scripts to see updated results.

## Understanding the Results

### Key Questions to Explore

1. **Does actual rationality matter more than beliefs?**
   - Compare scenarios with same beliefs but different actual β values

2. **How do mismatched beliefs affect cooperation?**
   - Compare scenarios where reality matches vs. contradicts beliefs

3. **Is there a dominant strategy?**
   - Look at cooperation probabilities across all scenarios

4. **Do rational agents cooperate less or more?**
   - Compare high β vs. low β scenarios

### Interpreting Plots

- **Higher cooperation probability**: Agent more likely to cooperate
- **Lower cooperation probability**: Agent more likely to defect
- **Near 0.5 probabilities**: Agent is uncertain/indifferent between options
- **Extreme probabilities (near 0 or 1)**: Agent has strong preference

## The Memo Model

### How It Works

The model uses Memo's probabilistic programming capabilities to implement recursive reasoning:

1. **Agent Model**: Each agent computes expected utilities for cooperating vs. defecting
2. **Opponent Model**: To compute expected utilities, agent must reason about opponent's likely action
3. **Recursive Reasoning**: The opponent model itself requires reasoning about what the opponent thinks you'll do
4. **Softmax Decision**: Final action chosen probabilistically based on utilities and β

### Key Memo Features Used

- `@memo` decorator: Defines probabilistic models
- `chooses()`: Specifies probabilistic choice points
- `wpp=`: Weighted probability for choices (softmax weights)
- `infers()`: Performs probabilistic inference
- JAX arrays: Efficient parallel computation

## Troubleshooting

### "Module 'memo' not found"
- Make sure you installed `memo-lang` not `memo`: `pip install memo-lang`
- Verify virtual environment is activated

### "simulation_results.json not found"
- Run `python prisoners_dilemma.py` before `python plot_results.py`

### JAX errors on Mac
- If you have an ARM Mac (M1/M2), ensure you installed ARM-compatible Python
- See [Memo FAQ](https://github.com/kach/memo#faq) for Mac-specific issues

### Numerical stability warnings
- If you see unexpected zeros or infinities, you may need to adjust the beta values
- Try values between 0.1 and 20 for stability

## Extensions and Further Work

### Possible Modifications

1. **Grid Search**: Vary β values continuously to see full cooperation landscape
2. **Different Beliefs**: Allow agents to have different beliefs about each other
3. **Asymmetric Payoffs**: Change payoff matrix to be different for each player
4. **Repeated Games**: Extend to multiple rounds (requires different Memo structure)
5. **More Players**: Extend to N-player prisoner's dilemma

### For Your Mini Project

Consider exploring:
- How does the cooperation probability change with β values in between 0.5 and 10?
- What happens with more extreme β values (0.1, 0.01, 50, 100)?
- How do results compare to human behavioral data?
- Can you predict which scenarios lead to mutual cooperation?

## References

1. Jara-Ettinger, J., Gweon, H., Schulz, L. E., & Tenenbaum, J. B. (2020). The Naive Utility Calculus as a unified, quantitative framework for action understanding. *Cognitive Psychology, 123*, 101334.

2. Memo Language Documentation: https://github.com/kach/memo

3. Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books.

4. Rapoport, A., & Chammah, A. M. (1965). *Prisoner's Dilemma*. University of Michigan Press.

## License

This code is provided for educational purposes as part of a computational cognitive science mini-project.

## Contact

For questions about Memo, see the [Memo GitHub repository](https://github.com/kach/memo) or consult the Handbook.pdf in the Memo repository.

---

**Note**: This implementation demonstrates recursive reasoning about rationality in strategic decision-making. The model captures how agents' beliefs about each other's rationality levels influence their own choices, even when those beliefs may not match reality.

