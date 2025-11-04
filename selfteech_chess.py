import chess
import chess.pgn
import random
import pickle
import os
import math

# hyperperamters
WEIGHTS_FILE = "weights.pkl"
NUM_GAMES = 100
EPSILON = 0.5
ALPHA = 0.0005
DISCOUNT = 1.0
MAX_MOVES = 600
CENTER_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5, chess.C4, chess.F4, chess.C5, chess.F5}
PIECE_VALUES = {chess.PAWN:1.0, chess.KNIGHT:3.0, chess.BISHOP:3.25, chess.ROOK:5.0, chess.QUEEN:9.0, chess.KING:0.0}

def extract_features(board):
    """Extract normalized features"""
    # material diff scaled
    mat_white = sum(len(board.pieces(pt, chess.WHITE)) * val for pt,val in PIECE_VALUES.items())
    mat_black = sum(len(board.pieces(pt, chess.BLACK)) * val for pt,val in PIECE_VALUES.items())
    mat_diff = (mat_white - mat_black)/10.0

    # mobility scaled
    turn_backup = board.turn
    board.turn = chess.WHITE
    mobility_white = board.legal_moves.count()/20.0
    board.turn = chess.BLACK
    mobility_black = board.legal_moves.count()/20.0
    board.turn = turn_backup
    mobility_diff = mobility_white - mobility_black

    # center control
    center_white = sum(1.0 if board.piece_at(sq) and board.piece_at(sq).color==chess.WHITE else 0.0 for sq in CENTER_SQUARES)/4.0
    center_black = sum(1.0 if board.piece_at(sq) and board.piece_at(sq).color==chess.BLACK else 0.0 for sq in CENTER_SQUARES)/4.0
    center_diff = center_white - center_black

    # pawn structure
    doubled_white = sum(max(0,sum(1 for r in range(8) if board.piece_at(chess.square(f,r)) and board.piece_at(chess.square(f,r)).piece_type==chess.PAWN and board.piece_at(chess.square(f,r)).color==chess.WHITE)-1) for f in range(8))/2.0
    doubled_black = sum(max(0,sum(1 for r in range(8) if board.piece_at(chess.square(f,r)) and board.piece_at(chess.square(f,r)).piece_type==chess.PAWN and board.piece_at(chess.square(f,r)).color==chess.BLACK)-1) for f in range(8))/2.0
    pawn_struct_diff = -(doubled_white - doubled_black)

    bias = 1.0
    return [mat_diff, mobility_diff, center_diff, pawn_struct_diff, bias]

# linaer
class LinearEvaluator:
    def __init__(self):
        if os.path.exists(WEIGHTS_FILE):
            self.load()
        else:
            self.w = [1.0, 0.1, 0.5, 0.5, 0.0]
            self.w = [wi + random.uniform(-0.05,0.05) for wi in self.w]

    def value(self, features):
        return sum(w*f for w,f in zip(self.w,features))

    def save(self):
        with open(WEIGHTS_FILE,"wb") as f:
            pickle.dump(self.w,f)

    def load(self):
        with open(WEIGHTS_FILE,"rb") as f:
            self.w = pickle.load(f)
        if len(self.w)!=5:
            self.w = [1.0,0.1,0.5,0.5,0.0]

# move choice
def choose_move(board, evaluator, epsilon=EPSILON):
    legal = list(board.legal_moves)
    if not legal:
        return None
    if random.random()<epsilon:
        return random.choice(legal)

    best_move = None
    best_val = -1e9 if board.turn==chess.WHITE else 1e9
    for mv in legal:
        board.push(mv)
        val = evaluator.value(extract_features(board))
        board.pop()
        if board.turn==chess.WHITE:
            if val>best_val:
                best_val=val
                best_move=mv
        else:
            if val<best_val:
                best_val=val
                best_move=mv
    return best_move if best_move else random.choice(legal)

# self play
def play_self_game(evaluator, epsilon=EPSILON, save_pgn_path="selfplay_games.pgn", game_index=None):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    states_feats = []
    moves_made = 0

    while not board.is_game_over(claim_draw=True) and moves_made<MAX_MOVES:
        mv = choose_move(board, evaluator, epsilon)
        if mv is None: break
        feats = extract_features(board)
        states_feats.append((feats, board.turn))
        board.push(mv)
        node = node.add_variation(mv)
        moves_made += 1
        print(board)
        print("Move:", mv.uci(), "\n")

    # result
    result = board.result(claim_draw=True)
    if result=="1-0": z=1.0
    elif result=="0-1": z=0.0
    else: z=0.5

    # PGN headers
    game.headers["Event"]="Self-Play Training"
    game.headers["White"]="SelfTeachEngine"
    game.headers["Black"]="SelfTeachEngine"
    game.headers["Result"]=result
    if game_index: game.headers["Round"]=str(game_index)

    with open(save_pgn_path,"a") as f:
        print(game,file=f,end="\n\n")
    print(f"Saved Game {game_index} to {save_pgn_path}\n")
    return states_feats, z, board

# td update
def td_update(evaluator, states_feats, z, alpha=ALPHA, discount=DISCOUNT):
    T = len(states_feats)
    if T==0: return
    V = [evaluator.value(feats) for feats,_ in states_feats]
    terminal_value = (z-0.5)*2.0
    for t in range(T-1,-1,-1):
        v_t = V[t]
        v_next = terminal_value if t==T-1 else V[t+1]
        delta = discount*v_next - v_t
        delta = max(min(delta,1.0),-1.0)
        feats = states_feats[t][0]
        for i in range(len(evaluator.w)):
            update = alpha*delta*feats[i]
            update = max(min(update,0.05),-0.05)
            evaluator.w[i]+=update
        for i in range(len(evaluator.w)):
            evaluator.w[i]*=0.9999

# train loop
def play_and_learn(num_games=NUM_GAMES, epsilon=EPSILON, alpha=ALPHA):
    evaluator = LinearEvaluator()
    stats = {"white":0,"black":0,"draw":0}
    for g in range(1,num_games+1):
        states_feats, z, final_board = play_self_game(evaluator, epsilon=epsilon, game_index=g)
        if z==1.0: stats["white"]+=1
        elif z==0.0: stats["black"]+=1
        else: stats["draw"]+=1
        td_update(evaluator, states_feats, z, alpha=alpha)
        print(f"[Game {g}/{num_games}] W:{stats['white']} B:{stats['black']} D:{stats['draw']}; weights: {[round(x,3) for x in evaluator.w]}\n")
    evaluator.save()
    print("Training complete. Final weights:", evaluator.w)
    return evaluator

if __name__=="__main__":
    print("Self-teaching chess engine with PGN logging and persistent learning")
    print("Options:\n 1) Train by self-play")
    choice = input("Choose 1 to train: ").strip()
    if choice=="1":
        ng = input(f"Number of self-play games [default {NUM_GAMES}]: ").strip()
        ng = int(ng) if ng else NUM_GAMES
        eps = input(f"Epsilon for exploration [default {EPSILON}]: ").strip()
        eps = float(eps) if eps else EPSILON
        alpha_in = input(f"Learning rate alpha [default {ALPHA}]: ").strip()
        alpha_in = float(alpha_in) if alpha_in else ALPHA
        print(f"Running {ng} games with eps={eps}, alpha={alpha_in} ...")
        play_and_learn(num_games=ng, epsilon=eps, alpha=alpha_in)
    else:
        print("Exiting.")
