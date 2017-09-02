import chess.pgn

path = "C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\150pgn.txt"
all_games_in_file = []
pgn_file_handle = open(path)

while True:
    game = chess.pgn.read_game(pgn_file_handle)
    if game == None:
        break
    else:
        all_games_in_file.append(game)
pass