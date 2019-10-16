import numpy as np
from evoframe.games import Game
from copy import deepcopy

class Tris(Game):
    """Player1 starts. Rewards of both players are returned."""
    PLAYER_1 = 1
    PLAYER_2 = -1
    EMPTY = 0
    DRAW = 0
    CONTINUE = 2
    PLAYERS = [PLAYER_1, PLAYER_2]

    def __init__(self):
        self.board = np.array([np.array([self.EMPTY for i in range(3)]) for j in range(3)])

    def get_available_actions(self):
        available_actions = []
        for ir,row in enumerate(self.board):
            for ic,cell in enumerate(row):
                if cell == self.EMPTY:
                    available_actions += [[1 if ir * 3 + ic == i else 0 for i in range(9)]]
        return available_actions

    def get_next_state(self, action):
        board = deepcopy(self.board)
        i_action = np.array(action).argmax()
        i_action_row, i_action_col = i_action // 3, i_action % 3
        board[i_action_row][i_action_col] = self.PLAYER_1
        return board

    def check_win(self):
        # check rows
        board = self.board
        for row in board:
            for player in self.PLAYERS:
                if np.all(np.equal(row, np.full(3, player))):
                    return player

        # check cols
        board = self.board.transpose()
        for row in board:
            for player in self.PLAYERS:
                if np.all(np.equal(row, np.full(3, player))):
                    return player

        # check diagonals
        diags = []
        diags.append(np.array([board[i][i] for i in range(3)]))
        diags.append(np.array([board[i][3 - i - 1] for i in range(3)]))
        for row in diags:
            for player in self.PLAYERS:
                if np.all(np.equal(row, np.full(3, player))):
                    return player

        # check draw
        exist_empty = False
        for row in self.board:
            for cell in row:
                if cell == self.EMPTY:
                    exist_empty = True
        if not exist_empty:
            return self.DRAW

        return self.CONTINUE

    def extract_move(self, prediction):
        highest_value = -100000
        highest_value_index = -1
        for i,pred in enumerate(prediction):
            if self.board[i//3][i%3] == self.EMPTY and pred > highest_value:
                highest_value = pred
                highest_value_index = i
        return highest_value_index

    def do_move(self, move, player):
        self.board[move//3][move%3] = player

    def opposite_board(self):
        return np.array([np.array([self.PLAYER_2 if self.board[row][col] == self.PLAYER_1
                                   else self.PLAYER_1 if self.board[row][col] == self.PLAYER_2
                                   else self.EMPTY for col in range(3)]) for row in range(3)])

    def play(self, agent_1, agent_2, interactive=False):
        player_turn = self.PLAYER_1

        if interactive:
            self.print_board()

        result = self.check_win()
        while result == self.CONTINUE:
            if player_turn == self.PLAYER_1:
                prediction = agent_1.predict(self)
            else:
                self_opposite = deepcopy(self)
                self_opposite.board = self_opposite.opposite_board()
                prediction = agent_2.predict(self_opposite)
            move = self.extract_move(prediction)
            self.do_move(move, player_turn)

            if interactive:
                self.print_board()

            if player_turn == self.PLAYER_1:
                player_turn = self.PLAYER_2
            else:
                player_turn = self.PLAYER_1

            result = self.check_win()

        if interactive:
                self.print_board()

        opponent_result = self.PLAYER_2 if result == self.PLAYER_1 else self.PLAYER_1 if result == self.PLAYER_2 else self.DRAW
        return result, opponent_result

    def print_board(self):
        for row in self.board:
            for cell in row:
                print(cell, end=" ")
            print("")
        print("-"*30)
