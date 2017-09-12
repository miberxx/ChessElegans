
import chess

board = chess.Board()
while not board.is_stalemate() and not board.is_insufficient_material() and not board.is_game_over():
    w_b_move = board.fen().split(' ')[1]
    print('---------------')
    print(board)
    print('                ')
    move_in = input(w_b_move + ' to move: ')
    if move_in == 'undo':
        board.pop()
        print(board)
    else:
        move  = chess.Move.from_uci(move_in)
        if move in board.legal_moves:
            board.push(move)
        else:
            print('illegal move')
        print('---------------')

if board.is_stalemate():
    print('1/2 1/2 stalemate')
    print(board)
elif board.is_insufficient_material():
    print('1/2 1/2 insufficient material')
    print(board)
else:
    print('Game over')
    print(board)