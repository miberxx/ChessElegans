import chess.pgn
#==============================================================================================================================================================
class params:
    RAW_INPUT_FILE = "C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\4.pgn"
    CELEGANS_INPUT_FILE = "C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\4_out.txt"
    READ_FILE_VERBOSE = False
    WRITE_VERIFIED_GAMES = False
    VERIFY_VERBOSE = False

def check_longest_game_in_file(inpath, intxtenc):
    print('Opening file for read ' + inpath + ' with encoding ' + intxtenc)
    pgn_input_file_handle = open(inpath, encoding=intxtenc)
    count_games = 0
    max_moves = 0
    print('Reading games and counting maximum moves...')
    while True:
        game = chess.pgn.read_game(pgn_input_file_handle)
        if game == None:
            break
        if game.errors == []:
            count_games += 1
            node = game
            move_list = []
            while not node.is_end():
                next_node = node.variations[0]
                move_list.append(next_node.move)
                node = next_node
        else:
            continue
        if len(move_list) > max_moves:
                max_moves = len(move_list)
    print(str(count_games) + ' games checked with max moves: ' + str(max_moves))
    return [count_games,max_moves]
#==============================================================================================================================================================
def create_CElegans_Input_File(inpath, intxtenc, outpath, outtxtenc):
    print('Opening file for read ' + inpath + ' with encoding ' + intxtenc)
    pgn_input_file_handle = open(inpath, encoding = intxtenc)
    print('opeing file for write ' + outpath + ' with encoding ' + outtxtenc)
    pgn_output_file_handle = open(outpath, 'w', encoding = outtxtenc)

    count_games = 0
    print('Reading games...')
    while True:
        game = chess.pgn.read_game(pgn_input_file_handle)
        if game == None:
            break
        if game.errors == []:
            count_games += 1
            node = game
            move_list = []
            while not node.is_end():
                next_node = node.variations[0]
                move_list.append(next_node.move)
                node = next_node
        else:
            continue

        board = chess.Board()

        for move in move_list:
            fen_nn = convert_board_FEN_NN(board.fen())
            move_nn = convert_move_UCI_NN(str(move))
            if len(str(move)) == 4:
                move_nn = move_nn + ',0,0,0,0'
            sample = fen_nn + '/' + move_nn + '\n'
            pgn_output_file_handle.write(sample)
            board.push(move)
            pass
    pgn_output_file_handle.close()
#==============================================================================================================================================================
def test_conversions(path, txtenc):
    all_games_in_file = []
    pgn_file_handle = open(path, encoding = txtenc)

    print('Reading game file...')
    count_games = 0
    while True:
        game = chess.pgn.read_game(pgn_file_handle)
        if not game == None:
            if params.WRITE_VERIFIED_GAMES:
                game.accept(exporter)
        if game == None:
            break
        else:
            all_games_in_file.append(game)
            count_games += 1
            if params.READ_FILE_VERBOSE:
                #print(str(count_games) + '. ' + str(game.headers))
                print(str(count_games))
    print('Number of games read: ' + str(count_games))
    print('Validating...')
    for game in all_games_in_file:
        if params.VERIFY_VERBOSE:
            print(str(game))
        node = game
        move_list = []
        while not node.is_end():
            next_node = node.variations[0]
            move_list.append(next_node.move)
            node = next_node
        board = chess.Board()
        if params.VERIFY_VERBOSE:
            print('----------------------------------------------------------------------------------------------------')
            print(board.fen())

        if not validate_conversion_board(board.fen()):
            print('Validation error for position ' + game)
            exit(0)
        for move in move_list:
            if params.VERIFY_VERBOSE:
                print(str(move))
            if not validate_conversion_move(str(move)):
                print('Validation error for move ' + str(move))
                exit(0)
            board.push(move)
            if not validate_conversion_board(board.fen()):
                print('Validation error for position ' + str(game))
                exit(0)
#==============================================================================================================================================================
def validate_conversion_board(fen):
    fen_split = fen.split(' ')
    fen_for_compare = ''.join(fen_split[0]) + ' ' + ''.join(fen_split[1]) + ' ' + ''.join(fen_split[2]) + ' ' + ''.join(fen_split[3])
    fen_converted_back = convert_board_NN_FEN(convert_board_FEN_NN(fen))
    if fen_for_compare == fen_converted_back:
        if params.VERIFY_VERBOSE:
            print('FEN original: ' + fen)
            print('FEN convert : ' + fen_converted_back)
        return True
    else:
        print('FEN original: ' + fen)
        print('FEN convert : ' + fen_converted_back)
        return False
#==============================================================================================================================================================
def validate_conversion_move(uci_move):
    tmp = convert_move_NN_UCI(convert_move_UCI_NN(uci_move))
    if uci_move == tmp:
        if params.VERIFY_VERBOSE:
            print('UCI original: ' + uci_move)
            print('UCI convert : ' + tmp)
        return True
    else:
        print('UCI original: ' + uci_move)
        print('UCI convert : ' + tmp)
        return False
# ==============================================================================================================================================================
def convert_board_FEN_NN(fen):

    tmp_fen = fen
    split = fen.split(' ')
    board = split[0]
    move_w_b = split[1]
    castle = split[2]
    en_pass = split[3]

#Board
    #Prepare Board zeros
    pos = 0
    for character in board:
        if character in ['1','2','3','4','5','6','7','8']:
            if character == '1':
                board = board[:pos] + '0' + board[pos+1:]
                pos = pos + 1
            if character == '2':
                board = board[:pos] + '00' + board[pos+1:]
                pos = pos + 2
            if character == '3':
                board = board[:pos] + '000' + board[pos+1:]
                pos = pos + 3
            if character == '4':
                board = board[:pos] + '0000' + board[pos+1:]
                pos = pos + 4
            if character == '5':
                board = board[:pos] + '00000' + board[pos+1:]
                pos = pos + 5
            if character == '6':
                board = board[:pos] + '000000' + board[pos+1:]
                pos = pos + 6
            if character == '7':
                board = board[:pos] + '0000000' + board[pos+1:]
                pos = pos + 7
            if character == '8':
                board = board[:pos] + '00000000' + board[pos+1:]
                pos = pos + 8
        else:
            pos = pos + 1

    #Process P/p
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
    #Process R/r
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
    #Process N/n
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
    #Process B/b
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
    #Process Q/q
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
    #Process K/k
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
#Move_w_b
    to_move = ''
    if move_w_b == 'w':
        to_move = '1'
    else:
        to_move = '-1'
#Castling
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
#EnPassant
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
def convert_move_UCI_NN(uci_move):
    #a,b,c,d,e,f,g,h
    if uci_move[0] == 'a':
        from_move_file = '1,0,0,0,0,0,0,0'
    if uci_move[0] == 'b':
        from_move_file = '0,1,0,0,0,0,0,0'
    if uci_move[0] == 'c':
        from_move_file = '0,0,1,0,0,0,0,0'
    if uci_move[0] == 'd':
        from_move_file = '0,0,0,1,0,0,0,0'
    if uci_move[0] == 'e':
        from_move_file = '0,0,0,0,1,0,0,0'
    if uci_move[0] == 'f':
        from_move_file = '0,0,0,0,0,1,0,0'
    if uci_move[0] == 'g':
        from_move_file = '0,0,0,0,0,0,1,0'
    if uci_move[0] == 'h':
        from_move_file = '0,0,0,0,0,0,0,1'
    #1,2,3,4,5,6,7,8
    if uci_move[1] == '1':
        from_move_rank = '1,0,0,0,0,0,0,0'
    if uci_move[1] == '2':
        from_move_rank = '0,1,0,0,0,0,0,0'
    if uci_move[1] == '3':
        from_move_rank = '0,0,1,0,0,0,0,0'
    if uci_move[1] == '4':
        from_move_rank = '0,0,0,1,0,0,0,0'
    if uci_move[1] == '5':
        from_move_rank = '0,0,0,0,1,0,0,0'
    if uci_move[1] == '6':
        from_move_rank = '0,0,0,0,0,1,0,0'
    if uci_move[1] == '7':
        from_move_rank = '0,0,0,0,0,0,1,0'
    if uci_move[1] == '8':
        from_move_rank = '0,0,0,0,0,0,0,1'
    #a,b,c,d,e,f,g,h
    if uci_move[2] == 'a':
        to_move_file = '1,0,0,0,0,0,0,0'
    if uci_move[2] == 'b':
        to_move_file = '0,1,0,0,0,0,0,0'
    if uci_move[2] == 'c':
        to_move_file = '0,0,1,0,0,0,0,0'
    if uci_move[2] == 'd':
        to_move_file = '0,0,0,1,0,0,0,0'
    if uci_move[2] == 'e':
        to_move_file = '0,0,0,0,1,0,0,0'
    if uci_move[2] == 'f':
        to_move_file = '0,0,0,0,0,1,0,0'
    if uci_move[2] == 'g':
        to_move_file = '0,0,0,0,0,0,1,0'
    if uci_move[2] == 'h':
        to_move_file = '0,0,0,0,0,0,0,1'
    #1,2,3,4,5,6,7,8
    if uci_move[3] == '1':
        to_move_rank = '1,0,0,0,0,0,0,0'
    if uci_move[3] == '2':
        to_move_rank = '0,1,0,0,0,0,0,0'
    if uci_move[3] == '3':
        to_move_rank = '0,0,1,0,0,0,0,0'
    if uci_move[3] == '4':
        to_move_rank = '0,0,0,1,0,0,0,0'
    if uci_move[3] == '5':
        to_move_rank = '0,0,0,0,1,0,0,0'
    if uci_move[3] == '6':
        to_move_rank = '0,0,0,0,0,1,0,0'
    if uci_move[3] == '7':
        to_move_rank = '0,0,0,0,0,0,1,0'
    if uci_move[3] == '8':
        to_move_rank = '0,0,0,0,0,0,0,1'

    if len(uci_move)>4:
        if uci_move[4] == 'q':
            promote = '1,0,0,0'
        if uci_move[4] == 'r':
            promote = '0,1,0,0'
        if uci_move[4] == 'b':
            promote = '0,0,1,0'
        if uci_move[4] == 'n':
            promote =  '0,0,0,1'

    final_move = from_move_file + ',' + from_move_rank + ',' + to_move_file + ',' + to_move_rank
    if len(uci_move)>4:
        final_move = final_move + ',' + promote
    return final_move
#==============================================================================================================================================================
def remove_zeros(fen8):
    istring = fen8
    pieces_pos = []
    for pos, char in enumerate(istring):
        if istring[pos] is not '0':
            pieces_pos.append(pos)
    tmp = []
    if len(pieces_pos) == 0:
        tmp.append('8')
    elif len(pieces_pos) == 8:
        tmp.append(istring)
    else:
        for n, pos in enumerate(pieces_pos):
            if pos == 0:
                tmp.append(istring[pos])
            elif pos > 0 and istring[pos - 1] is not '0':
                tmp.append(istring[pos])
            else:
                if n == 0:
                    tmp.append(str(pos))
                    tmp.append(istring[pos])
                else:
                    tmp.append(str(pos - 1 - pieces_pos[n - 1]))
                    tmp.append(istring[pos])
        if 7 - pieces_pos[n] > 0:
            tmp.append(str(7 - pieces_pos[n]))
    final = ''.join(tmp)
    return final
# ==============================================================================================================================================================
def convert_board_NN_FEN(nn_board):
    board = nn_board.split(',')
    #fen 0-63 board, 64 move, 65-68 castling, 69-70 en passant
    fen = list('0000000000000000000000000000000000000000000000000000000000000000')
    P = board[0:64]
    R = board[64:128]
    N = board[128:192]
    B = board[192:256]
    Q = board[256:320]
    K = board[320:384]
    to_move = board[384:385]
    castling = board[385:389]
    en_passant = board[389:405]
    #Find the Ps
    pos = 0
    for pos, pawn in enumerate(P):
        if pawn == '1':
            fen[pos] = 'P'
            continue
        elif pawn == '-1':
            fen[pos] = 'p'
    #Find the Rs
    pos = 0
    for pos, rook in enumerate(R):
        if rook == '1':
            fen[pos] = 'R'
            continue
        elif rook == '-1':
            fen[pos] = 'r'
    pos = 0
    for pos, knight in enumerate(N):
        if knight == '1':
            fen[pos] = 'N'
            continue
        elif knight == '-1':
            fen[pos] = 'n'
    pos = 0
    for pos, bishop in enumerate(B):
        if bishop == '1':
            fen[pos] = 'B'
            continue
        elif bishop == '-1':
            fen[pos] = 'b'
    pos = 0
    for pos, queen in enumerate(Q):
        if queen == '1':
            fen[pos] = 'Q'
            continue
        elif queen == '-1':
            fen[pos] = 'q'
    pos = 0
    for pos, king in enumerate(K):
        if king == '1':
            fen[pos] = 'K'
            continue
        elif king == '-1':
            fen[pos] = 'k'
    #to move
    if to_move[0] == '1':
        fen_to_move = 'w'
    else:
        fen_to_move = 'b'


    board_1 = fen[0:8]
    board_2 = fen[8:16]
    board_3 = fen[16:24]
    board_4 = fen[24:32]
    board_5 = fen[32:40]
    board_6 = fen[40:48]
    board_7 = fen[48:56]
    board_8 = fen[56:64]

    board_nn_fen = remove_zeros(''.join(board_1)) + '/' + remove_zeros(''.join(board_2)) + '/' + remove_zeros(''.join(board_3)) + '/' + remove_zeros(''.join(board_4)) + '/' + remove_zeros(''.join(board_5)) + '/' + remove_zeros(''.join(board_6)) + '/' + remove_zeros(''.join(board_7)) + '/' + remove_zeros(''.join(board_8))
    pass

    #castling
    castling_fen = []
    if castling[0] == '1':
        castling_fen.append('K')
    if castling[1] == '1':
        castling_fen.append('Q')
    if castling[2] == '1':
        castling_fen.append('k')
    if castling[3] == '1':
        castling_fen.append('q')
    if castling[0] == '0' and castling[1] == '0' and castling[2] == '0' and castling[3] == '0':
        castling_fen.append('-')
    #en_passant
    en_passant_fen = '-'
    if en_passant[0] == '1':
        en_passant_fen = 'a3'
    if en_passant[1] == '1':
        en_passant_fen = 'b3'
    if en_passant[2] == '1':
        en_passant_fen = 'c3'
    if en_passant[3] == '1':
        en_passant_fen = 'd3'
    if en_passant[4] == '1':
        en_passant_fen = 'e3'
    if en_passant[5] == '1':
        en_passant_fen = 'f3'
    if en_passant[6] == '1':
        en_passant_fen = 'g3'
    if en_passant[7] == '1':
        en_passant_fen = 'h3'
    if en_passant[8] == '1':
        en_passant_fen = 'a6'
    if en_passant[9] == '1':
        en_passant_fen = 'b6'
    if en_passant[10] == '1':
        en_passant_fen = 'c6'
    if en_passant[11] == '1':
        en_passant_fen = 'd6'
    if en_passant[12] == '1':
        en_passant_fen = 'e6'
    if en_passant[13] == '1':
        en_passant_fen = 'f6'
    if en_passant[14] == '1':
        en_passant_fen = 'g6'
    if en_passant[15] == '1':
        en_passant_fen = 'h6'



    final_fen = board_nn_fen + ' ' + ''.join(fen_to_move) + ' ' + ''.join(castling_fen) + ' ' + ''.join(en_passant_fen)
    return final_fen
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
    final_uci = file_from_uci+rank_from_uci+file_to_uci+rank_to_uci
    if len(nn_move) > 32:
        final_uci = final_uci + promote_uci
    return final_uci

#test_conversions(params.RAW_INPUT_FILE, 'ansi')
#check_longest_game_in_file(params.RAW_INPUT_FILE,'ansi')
create_CElegans_Input_File(params.RAW_INPUT_FILE,'ansi',params.CELEGANS_INPUT_FILE,'utf-8')