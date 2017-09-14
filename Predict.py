import chess
import numpy as np
from keras.models import model_from_json

class Params:
    CHECKPOINT_FILE_WEIGHTS = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\ModelRun\\weights.best.hdf5'
    CHECKPOINT_FILE_MODEL = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\ModelRun\\model.json'
    TEST_GAMES_FILE = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\ModelRun\\1123testgames.pgn'
#==============================================================================================================================================================
def load_model():
    json_file = open(Params.CHECKPOINT_FILE_MODEL, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(Params.CHECKPOINT_FILE_WEIGHTS)
    return loaded_model
#==============================================================================================================================================================
def predict(model,fen):
    tmp = convert_board_FEN_NN(fen)
    tmp = tmp.rstrip()
    tmp = tmp.replace('\n', '')
    fen_nn = list(map(int, tmp.split(',')))
    x_nn = np.array(fen_nn).reshape(1,405)
    tmp = model.predict(x_nn,1,verbose=0)
    top3_file_from, top3_rank_from, top3_file_to, top3_rank_to, top_promotion = convert_prediction_to_NN(tmp)
    #to do create 3 moves
    tmp1_file_from = ','.join(str(i) for i in top3_file_from[0])
    tmp2_file_from = ','.join(str(i) for i in top3_file_from[1])
    tmp3_file_from = ','.join(str(i) for i in top3_file_from[2])

    tmp1_rank_from = ','.join(str(i) for i in top3_rank_from[0])
    tmp2_rank_from = ','.join(str(i) for i in top3_rank_from[1])
    tmp3_rank_from = ','.join(str(i) for i in top3_rank_from[2])

    tmp1_file_to = ','.join(str(i) for i in top3_file_to[0])
    tmp2_file_to = ','.join(str(i) for i in top3_file_to[1])
    tmp3_file_to = ','.join(str(i) for i in top3_file_to[2])

    tmp1_rank_to = ','.join(str(i) for i in top3_rank_to[0])
    tmp2_rank_to = ','.join(str(i) for i in top3_rank_to[1])
    tmp3_rank_to = ','.join(str(i) for i in top3_rank_to[2])

    promote1 = ','.join(str(i) for i in top_promotion)

    move1 = tmp1_file_from + ',' + tmp1_rank_from + ',' + tmp1_file_to + ',' + tmp1_rank_to + ',' + promote1
    move2 = tmp2_file_from + ',' + tmp2_rank_from + ',' + tmp2_file_to + ',' + tmp2_rank_to + ',' + '0,0,0,0'
    move3 = tmp3_file_from + ',' + tmp3_rank_from + ',' + tmp3_file_to + ',' + tmp3_rank_to + ',' + '0,0,0,0'
    moves = []
    moves.append(move1)
    moves.append(move2)
    moves.append(move3)
    y_UCI = []
    for i in range(0,len(moves)):
        y_UCI.append(convert_move_NN_UCI(moves[i]))
    return y_UCI
#==============================================================================================================================================================
def convert_board_FEN_NN(fen):
    tmp_fen = fen
    split = fen.split(' ')
    board = split[0]
    move_w_b = split[1]
    castle = split[2]
    en_pass = split[3]

    # Board
    # Prepare Board zeros
    pos = 0
    for character in board:
        if character in ['1', '2', '3', '4', '5', '6', '7', '8']:
            if character == '1':
                board = board[:pos] + '0' + board[pos + 1:]
                pos = pos + 1
            if character == '2':
                board = board[:pos] + '00' + board[pos + 1:]
                pos = pos + 2
            if character == '3':
                board = board[:pos] + '000' + board[pos + 1:]
                pos = pos + 3
            if character == '4':
                board = board[:pos] + '0000' + board[pos + 1:]
                pos = pos + 4
            if character == '5':
                board = board[:pos] + '00000' + board[pos + 1:]
                pos = pos + 5
            if character == '6':
                board = board[:pos] + '000000' + board[pos + 1:]
                pos = pos + 6
            if character == '7':
                board = board[:pos] + '0000000' + board[pos + 1:]
                pos = pos + 7
            if character == '8':
                board = board[:pos] + '00000000' + board[pos + 1:]
                pos = pos + 8
        else:
            pos = pos + 1

    # Process P/p
    P = ''
    for square in board:
        if square == '/':
            continue
        elif square == 'P':
            P = P + '1' + ','
        elif square == 'p':
            P = P + '-1' + ','
        else:
            P = P + '0' + ','
    P = P[:-1]
    # Process R/r
    R = ''
    for square in board:
        if square == '/':
            continue
        elif square == 'R':
            R = R + '1' + ','
        elif square == 'r':
            R = R + '-1' + ','
        else:
            R = R + '0' + ','
    R = R[:-1]
    # Process N/n
    N = ''
    for square in board:
        if square == '/':
            continue
        elif square == 'N':
            N = N + '1' + ','
        elif square == 'n':
            N = N + '-1' + ','
        else:
            N = N + '0' + ','
    N = N[:-1]
    # Process B/b
    B = ''
    for square in board:
        if square == '/':
            continue
        elif square == 'B':
            B = B + '1' + ','
        elif square == 'b':
            B = B + '-1' + ','
        else:
            B = B + '0' + ','
    B = B[:-1]
    # Process Q/q
    Q = ''
    for square in board:
        if square == '/':
            continue
        elif square == 'Q':
            Q = Q + '1' + ','
        elif square == 'q':
            Q = Q + '-1' + ','
        else:
            Q = Q + '0' + ','
    Q = Q[:-1]
    # Process K/k
    K = ''
    for square in board:
        if square == '/':
            continue
        elif square == 'K':
            K = K + '1' + ','
        elif square == 'k':
            K = K + '-1' + ','
        else:
            K = K + '0' + ','
    K = K[:-1]
    # Move_w_b
    to_move = ''
    if move_w_b == 'w':
        to_move = '1'
    else:
        to_move = '-1'
        # Castling
    castling = '0000'
    if 'K' in castle:
        castling = '1' + castling[1:]
    if 'Q' in castle:
        castling = castling[0] + '1' + castling[2:]
    if 'k' in castle:
        castling = castling[0:2] + '1' + castling[3:]
    if 'q' in castle:
        castling = castling[0:3] + '1'
    tmp = list(castling)
    final_castling = tmp[0] + ',' + tmp[1] + ',' + tmp[2] + ',' + tmp[3]
    # EnPassant
    en_passant = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0'
    if en_pass == 'a3':
        en_passant = '1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0'
    elif en_pass == 'b3':
        en_passant = '0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0'
    elif en_pass == 'c3':
        en_passant = '0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0'
    elif en_pass == 'd3':
        en_passant = '0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0'
    elif en_pass == 'e3':
        en_passant = '0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0'
    elif en_pass == 'f3':
        en_passant = '0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0'
    elif en_pass == 'g3':
        en_passant = '0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0'
    elif en_pass == 'h3':
        en_passant = '0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0'
    elif en_pass == 'a6':
        en_passant = '0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0'
    elif en_pass == 'b6':
        en_passant = '0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0'
    elif en_pass == 'c6':
        en_passant = '0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0'
    elif en_pass == 'd6':
        en_passant = '0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0'
    elif en_pass == 'e6':
        en_passant = '0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0'
    elif en_pass == 'f6':
        en_passant = '0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0'
    elif en_pass == 'g6':
        en_passant = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0'
    elif en_pass == 'h6':
        en_passant = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1'

    position = P + ',' + R + ',' + N + ',' + B + ',' + Q + ',' + K + ',' + to_move + ',' + final_castling + ',' + en_passant
    return position
#==============================================================================================================================================================
def convert_move_NN_UCI(nn_move):
    nn_move = nn_move.split(',')
    file_from = nn_move[0:8]
    rank_from = nn_move[8:16]
    file_to = nn_move[16:24]
    rank_to = nn_move[24:32]

    if len(nn_move) > 32:
        promote = nn_move[32:36]

    for pos, value in enumerate(file_from):
        if value == '1':
            if pos == 0:
                file_from_uci = 'a'
                break
            elif pos == 1:
                file_from_uci = 'b'
                break
            elif pos == 2:
                file_from_uci = 'c'
                break
            elif pos == 3:
                file_from_uci = 'd'
                break
            elif pos == 4:
                file_from_uci = 'e'
                break
            elif pos == 5:
                file_from_uci = 'f'
                break
            elif pos == 6:
                file_from_uci = 'g'
                break
            elif pos == 7:
                file_from_uci = 'h'
                break
    for pos, value in enumerate(rank_from):
        if value == '1':
            if pos == 0:
                rank_from_uci = '1'
                break
            elif pos == 1:
                rank_from_uci = '2'
                break
            elif pos == 2:
                rank_from_uci = '3'
                break
            elif pos == 3:
                rank_from_uci = '4'
                break
            elif pos == 4:
                rank_from_uci = '5'
                break
            elif pos == 5:
                rank_from_uci = '6'
                break
            elif pos == 6:
                rank_from_uci = '7'
                break
            elif pos == 7:
                rank_from_uci = '8'
                break
    for pos, value in enumerate(file_to):
        if value == '1':
            if pos == 0:
                file_to_uci = 'a'
                break
            elif pos == 1:
                file_to_uci = 'b'
                break
            elif pos == 2:
                file_to_uci = 'c'
                break
            elif pos == 3:
                file_to_uci = 'd'
                break
            elif pos == 4:
                file_to_uci = 'e'
                break
            elif pos == 5:
                file_to_uci = 'f'
                break
            elif pos == 6:
                file_to_uci = 'g'
                break
            elif pos == 7:
                file_to_uci = 'h'
                break
    for pos, value in enumerate(rank_to):
        if value == '1':
            if pos == 0:
                rank_to_uci = '1'
                break
            elif pos == 1:
                rank_to_uci = '2'
                break
            elif pos == 2:
                rank_to_uci = '3'
                break
            elif pos == 3:
                rank_to_uci = '4'
                break
            elif pos == 4:
                rank_to_uci = '5'
                break
            elif pos == 5:
                rank_to_uci = '6'
                break
            elif pos == 6:
                rank_to_uci = '7'
                break
            elif pos == 7:
                rank_to_uci = '8'
                break
    if len(nn_move) > 32:
        for pos, value in enumerate(promote):
            if value == '1':
                if pos == 0:
                    promote_uci = 'q'
                    break
                if pos == 1:
                    promote_uci = 'r'
                    break
                if pos == 2:
                    promote_uci = 'b'
                    break
                if pos == 3:
                    promote_uci = 'n'
                    break
            else:
                promote_uci = ''
    final_uci = file_from_uci+rank_from_uci+file_to_uci+rank_to_uci
    if len(nn_move) > 32:
        final_uci = final_uci + promote_uci
    return final_uci
#==============================================================================================================================================================
def convert_prediction_to_NN(prediction):
    file_from = prediction[0][:8]
    rank_from = prediction[0][8:16]
    file_to = prediction[0][16:24]
    rank_to = prediction[0][24:32]
    promotion = prediction[0][32:36]

    top3_file_from = get_top_3(file_from)
    top3_rank_from = get_top_3(rank_from)
    top3_file_to = get_top_3(file_to)
    top3_rank_to = get_top_3(rank_to)

    median = np.median(promotion)
    max = 0
    max_index = None
    top_promotion = []
    for index, value in enumerate(promotion):
        if value > 10*median:
            top_promotion.append(1)
        else:
            top_promotion.append(0)
    return top3_file_from, top3_rank_from, top3_file_to, top3_rank_to, top_promotion
#==============================================================================================================================================================
def get_top_3(input):
    top_3 = []
    max = 0
    max_index = None
    tmp = []
    for index,value in enumerate(input):
        if value > max:
            max = value
            max_index = index
    for index, value in enumerate(input):
        if index == max_index:
            tmp.append(1)
            input[index] = 0
        else:
            tmp.append(0)
    top_3.append(tmp)

    max = 0
    max_index = None
    tmp = []
    for index, value in enumerate(input):
        if value > max:
            max = value
            max_index = index
    for index, value in enumerate(input):
        if index == max_index:
            tmp.append(1)
            input[index] = 0
        else:
            tmp.append(0)
    top_3.append(tmp)

    max = 0
    max_index = None
    tmp = []
    for index, value in enumerate(input):
        if value > max:
            max = value
            max_index = index
    for index, value in enumerate(input):
        if index == max_index:
            tmp.append(1)
            input[index] = 0
        else:
            tmp.append(0)
    top_3.append(tmp)

    return top_3
#==============================================================================================================================================================
np.random.seed(0)
model = load_model()


board = chess.Board()
while not board.is_stalemate() and not board.is_insufficient_material() and not board.is_game_over():
    w_b_move = board.fen().split(' ')[1]
    print('---------------')
    print(board)
    print('                ')
    if w_b_move == 'w':
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
    else:
        b_moves = predict(model,board.fen())
        move = chess.Move.from_uci(b_moves[0])
        if move in board.legal_moves:
            board.push(move)
        else:
            print('illegal move' + b_moves[0])

if board.is_stalemate():
    print('1/2 1/2 stalemate')
    print(board)
elif board.is_insufficient_material():
    print('1/2 1/2 insufficient material')
    print(board)
else:
    print('Game over')
    print(board)