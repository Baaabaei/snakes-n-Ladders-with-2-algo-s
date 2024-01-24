import numpy as np
import random

# Game settings
num_squares = 100
max_turns = 500

# Create board
board = {i: i for i in range(1, num_squares + 1)}
snakes = {16: 6, 48: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95: 75, 98: 78}
ladders = {3: 38, 8: 14, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}

for snake in snakes:
    board[snake] = snakes[snake]
for ladder in ladders:
    board[ladder] = ladders[ladder]

# Get algorithm choice   
print("Choose algorithm:")
print("1. Breadth-First Search")
print("2. Q-Learning")
choice = input("Enter 1 or 2: ")

if choice == '1':
    # BFS

    print("Playing with BFS...")
    from collections import deque

    # Game settings
    num_squares = 100
    max_turns = 500

    # Create board
    board = {i: i for i in range(1, num_squares + 1)}
    snakes = {96: 42, 94: 71, 75: 32, 47: 16, 25: 10, 37: 3}
    ladders = {54: 88, 41: 79, 22: 58, 14: 55, 12: 50, 4: 56}

    for snake in snakes:
        board[snake] = snakes[snake]
    for ladder in ladders:
        board[ladder] = ladders[ladder]

    # BFS    
    queue = deque([(1, 0)])  # square, turns
    visited = set()

    while queue:

        square, turns = queue.popleft()

        if square == 100:
            print("Finished in {} turns".format(turns))
            break

        if square in visited:
            continue

        visited.add(square)

        for die_roll in range(1, 7):
            new_square = square + die_roll
            if new_square > 100:
                new_square = 100
            new_square = board[new_square]
            queue.append((new_square, turns + 1))

    print("Explored {} squares".format(len(visited)))
    print("Explored", len(visited), "squares")

elif choice == '2':

    # Game settings
    num_squares = 100  # Number of squares on the board
    max_turn = 500  # Maximum number of turns before game ends

    # Create board with snakes and ladders
    board = np.arange(1, num_squares + 1)
    snakes = {96: 42, 94: 71, 75: 32, 47: 16, 25: 10, 37: 3}
    ladders = {54: 88, 41: 79, 22: 58, 14: 55, 12: 50, 4: 56}

    for snake in snakes:
        board[snake - 1] = snakes[snake]
    for ladder in ladders:
        board[ladder - 1] = ladders[ladder]

    # Q-learning parameters
    learning_rate = 0.1
    discount_factor = 0.6
    exploration = 0.1

    # Initialize Q-values
    Q = np.zeros((num_squares, 6))

    for episode in range(1000):

        turn = 0
        square = 1

        while square < num_squares and turn <= max_turn:

            # Choose action
            if random.uniform(0, 1) < exploration:
                action = np.random.randint(1, 7)  # Random move
            else:
                action = np.argmax(Q[square - 1] + np.random.randn(1, 6) * (1. / (episode + 1)))

            # Take action and get reward
            turn += 1
            square += action
            if square > num_squares: square = num_squares

            reward = 0
            if square in snakes:
                reward = -1
                square = snakes[square]
            elif square in ladders:
                reward = 1
                square = ladders[square]

            # Update Q-value
            max_Q = np.max(Q[square - 1])
            Q[square - 1, action - 1] += learning_rate * (reward +
                                                          discount_factor * max_Q - Q[square - 1, action - 1])

    print("Q-values trained!")

    # Play game with trained Q-values
    square = 1
    turn = 0

    while square < num_squares and turn <= max_turn:

        # Pick best action
        action = np.argmax(Q[square - 1]) + 1

        # Take action
        turn += 1
        square += action
        if square > num_squares: square = num_squares

        print("Turn:", turn, "Square:", square)
        print(board[0:square])

        if square in snakes:
            square = snakes[square]
            print("Bitten by snake!")
        elif square in ladders:
            square = ladders[square]
            print("Climbed ladder!")

    print("Game over!")
    print("Finished in", turns, "turns")

else:
    print("Invalid choice")