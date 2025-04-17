# Checkers AI - Monte Carlo Tree Search and Enhancements

## Overview
This project investigates the application of Monte Carlo Tree Search (MCTS) and its enhanced variants—RAVE-MC and Heuristic RAVE-MC—on the game of Checkers. The study compares four AI models including a baseline Minimax algorithm to evaluate decision-making efficiency, strategic strength, and adaptability in gameplay. After 546 simulated matches, the standard MCTS model achieved the highest performance with a 58% win rate. This project was conducted as a final project for a graduate Artificial Intelligence course at Fordham University.

## Requirements
- The only requirement is **Jupyter Notebook**.
- No additional dependencies need to be installed beyond standard Python libraries.
- Packages used:
  - pygame
  - numpy
  - time
  - random
  - copy

## Running the Notebook
1. Open Jupyter Notebook.
2. Load the provided notebook.
3. Update the CSV file paths in the notebook to the correct file location on your machine:
   ```python
   # Update these lines
   checkers_games = pd.read_csv('/Users/ssingson/Desktop/Checkers Games3.csv')
   checkers_games.to_csv('/Users/ssingson/Desktop/Checkers Games3.csv')
   ```
4. To test or run specific AI model matchups, call the main function with appropriate arguments:
   - showpygame: Set to True for a visual representation of the game (required if human player so they can play)
   - player: sets player to a model or a human player 
      - Player options: `'human'`, `'minimax'`, `'mcts'`, `'rave'`, `'heuristic'`
   - beta: For RAVE-MC models, hyperparamter to determine when to exploit a winning node versus exploring less explored nodes 
   - exploration_factor: hyperparameter to determine which similated nodes to explore based on percent of games the node was node was previously used; higher factor the more likely an unexplored node will be searched  
   - minimax_moves: Sets the number of moves in the future the white player model looks through to make best decision
   - mcts_iterations: number of times the model will run a specific end node before making a decision

## Key Features
- **AI Models Tested:**
  - Minimax
  - Monte Carlo Tree Search (MCTS)
  - Rapid Action Value Estimation MCTS (RAVE-MC)
  - Heuristic RAVE-MC
- **Game Engine:**
  - Checkers environment built using PyGame with custom rules and logic
- **Evaluation Metrics:**
  - Win/draw/loss rate
  - Game length (number of moves)
  - Number of kings and remaining pieces
  - Decision latency per move

## Results
- **Standard MCTS** was the top-performing model with a **58% win rate**, the fastest average win time, and strong defensive play.
- Heuristic RAVE-MC was also a strong performer, outperforming other models in matchups against MCTS.
- Secondary metrics highlighted how models varied in strategy, including kinging behavior and average moves per game.

## Acknowledgments
This project was developed by **Seth Singson-Robbins** at **Fordham University**.  
For further details, contact [seth.singson@gmail.com](mailto:seth.singson@gmail.com).
