import chess.pgn

RAW_INPUT_FILE = "C:\\Users\\michael\\Desktop\\ChessElegans\\HQ.pgn"
OUTPUT_FILE = "C:\\Users\\michael\\Desktop\\ChessElegans\\1K.pgn"
SPLIT_NO_GAMES = 1000

def split_CElegans_Input_File(inpath, intxtenc, outpath, outtxtenc):
    print('Opening file for read ' + inpath + ' with encoding ' + intxtenc)
    pgn_input_file_handle = open(inpath, encoding = intxtenc)
    print('opeing file for write ' + outpath + ' with encoding ' + outtxtenc)
    pgn_output_file_handle = open(outpath, 'w', encoding = outtxtenc)

    count_games = 1
    print('----------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Reading games...')
    while count_games <= SPLIT_NO_GAMES:
        game = chess.pgn.read_game(pgn_input_file_handle)
        if game == None:
            break
        if game.errors == []:
            count_games += 1
            #pgn_output_file_handle.write(game)
            print(game, file=pgn_output_file_handle , end="\n\n")
            print(str(SPLIT_NO_GAMES) + '/' + str(count_games-1))
        else:
            continue
    pgn_output_file_handle.close()
    print('----------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Games read: ' + str(count_games-1))


split_CElegans_Input_File(RAW_INPUT_FILE,'ansi',OUTPUT_FILE,'utf-8')