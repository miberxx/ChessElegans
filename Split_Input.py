import chess.pgn

RAW_INPUT_FILE = "C:\\Users\\michael\\Desktop\\ChessElegans\\HQ.pgn"
OUTPUT_FILE_PATH = "C:\\Users\\michael\\Desktop\\ChessElegans\\"
SPLIT_START = 1
SPLIT_END = 50000
SPLIT_CONSECUTIVE = False
SPLIT_INTERVAL = 100
SPLIT_TOTAL_GAMES = 1000
#==============================================================================================================================================================
def split_CElegans_Input_File(inpath, intxtenc, outpath, outtxtenc):
    print('Opening file for read ' + inpath + ' with encoding ' + intxtenc)
    pgn_input_file_handle = open(inpath, encoding = intxtenc)
    file_name = str(SPLIT_START) + '_' + str(SPLIT_END) + '.pgn'
    out_file = outpath + file_name
    print('opeing file for write ' + out_file + ' with encoding ' + outtxtenc)
    pgn_output_file_handle = open(out_file, 'w', encoding = outtxtenc)

    count_games = 0
    print('----------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Reading games...')
    while count_games <= SPLIT_END:
        game = chess.pgn.read_game(pgn_input_file_handle)
        if game == None:
            break
        if game.errors == []:
            count_games += 1
            #pgn_output_file_handle.write(game)
            if count_games >= SPLIT_START and count_games <= SPLIT_END:
                print(game, file=pgn_output_file_handle , end="\n\n")
                print(str(SPLIT_START)+'-'+str(SPLIT_END) + ' / ' + str(count_games))
        else:
            continue
    pgn_output_file_handle.close()
    print('----------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Games read: ' + str(count_games-1))
    print('Games written: ' + str(SPLIT_END - SPLIT_START +1))
#==============================================================================================================================================================
def split_consecutive(inpath, intxtenc, outpath, outtxtenc):
    print('Opening file for read ' + inpath + ' with encoding ' + intxtenc)
    pgn_input_file_handle = open(inpath, encoding=intxtenc)
    no_files = int(SPLIT_TOTAL_GAMES/SPLIT_INTERVAL)+1

    for n in range(0, no_files,1):
        pass


#==============================================================================================================================================================
if not SPLIT_CONSECUTIVE:
    split_CElegans_Input_File(RAW_INPUT_FILE,'ansi',OUTPUT_FILE_PATH,'utf-8')
else:
    split_consecutive(RAW_INPUT_FILE,'ansi',OUTPUT_FILE_PATH,'utf-8')