
import chess.pgn

class Params:

    PGN_INPUT_FILE_PATH = "C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\test_pgn.txt"

class ReadPGN:

    def __init__(self):
        pass

with open("C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\test_pgn.txt") as pgn:
    current_game = chess.pgn.read_game(pgn)
    node = current_game
    move_list = []
    while not node.is_end():

        next_node = node.variations[0]
        move_list.append(next_node.move)
        node = next_node




