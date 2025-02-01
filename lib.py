# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Imports
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# for evauluating hands
import eval7 as ev7 

# display 
from tqdm import tqdm 
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# data manipulation
from collections import defaultdict
import numpy as np
import joblib

# concurrency
import multiprocessing

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Constants
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Equity Methods
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# formatting/conversions

def ev7_hand_of_str_hand(str_hand):
    return [ev7.Card(str_hand[0:2]), ev7.Card(str_hand[2:4])]

def str_hand_of_ev7_hand(ev7_hand):
    rank0, suit0 = ev7_hand[0].rank, ev7_hand[0].suit
    rank1, suit1 = ev7_hand[1].rank, ev7_hand[1].suit
    if (rank0, suit0) <= (rank1, suit1):
        return f"{RANKS[rank1]}{SUITS[suit1]}{RANKS[rank0]}{SUITS[suit0]}"
    else:
        return f"{RANKS[rank0]}{SUITS[suit0]}{RANKS[rank1]}{SUITS[suit1]}"

def ev7_board_of_str_board(str_board):
    if len(str_board) == 3:
        return [ev7.Card(str_board[0:2]), ev7.Card(str_board[2:4]), ev7.Card(str_board[4:6])]  
    else:
        return [ev7.Card(str_board[0:2]), ev7.Card(str_board[2:4]), ev7.Card(str_board[4:6],)]  

def str_board_of_ev7_board(ev7_board):
    sorted_board = sorted(ev7_board, key=lambda card: (card.rank, card.suit), reverse=True)
    return "".join(f"{RANKS[card.rank]}{SUITS[card.suit]}" for card in sorted_board)

def sa_str_hand_of_str_hand(str_hand):
    rank0, suit0, rank1, suit1 = str_hand[0], str_hand[1], str_hand[2], str_hand[3]
    if rank0 == rank1: 
        return f"{rank0}{rank1}"
    elif suit0 == suit1:
        return f"{rank0}{rank1}s"
    else: 
        return f"{rank0}{rank1}o"

# preflop equity
    
def calculate_preflop_equity_monte_carlo(hero_hands: list[str], num_iterations: int) -> dict:

    raw_equity_dict = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'doubled_wins': 0}))
    deck = ev7.Deck()

    n_hero_hands = len(hero_hands)
    doc_str = f"Hero-hand{'s ' if n_hero_hands > 1 else ' '}"
    doc_str += ", ".join(f"{hand}" for hand in hero_hands[:-1])
    if n_hero_hands > 1:
        doc_str += f" and {hero_hands[-1]}"

    print(doc_str)

    ev7_and_str_hhands = [(ev7_hand_of_str_hand(str_hhand), str_hhand) for str_hhand in hero_hands]

    for _ in tqdm(range(num_iterations), "Monte-carlo equity estimation"):

        ev7_vhand_and_board = deck.sample(7)
        ev7_vhand_and_board_set = set(ev7_vhand_and_board)
        ev7_board = ev7_vhand_and_board[2:7]
        str_vhand = str_hand_of_ev7_hand(ev7_vhand_and_board[0:2])
        sa_str_vhand = sa_str_hand_of_str_hand(str_vhand)

        v_value = ev7.evaluate(ev7_vhand_and_board)

        for ev7_hhand, str_hhand in ev7_and_str_hhands: 

            # reject sample if there is a card repetition
            if any(ev7_card in ev7_vhand_and_board_set for ev7_card in ev7_hhand):
                continue 

            # increment hhand vs vhand count
            raw_equity_dict[str_hhand][sa_str_vhand]['n'] += 1

            # increment hhand vs vhand wins
            h_value = ev7.evaluate(ev7_hhand + ev7_board)
            if h_value > v_value:
                raw_equity_dict[str_hhand][sa_str_vhand]['doubled_wins'] += 2
            elif h_value == v_value:
                raw_equity_dict[str_hhand][sa_str_vhand]['doubled_wins'] += 1

                
    output_equity_dict = {}

    for str_hhand, all_vhands_equity_dict in raw_equity_dict.items():
        sa_str_hhand = sa_str_hand_of_str_hand(str_hhand)
        output_equity_dict[sa_str_hhand] = {}
        for sa_str_vhand, vhand_equity_dict in all_vhands_equity_dict.items():
            n = vhand_equity_dict['n']
            equity = (vhand_equity_dict['doubled_wins'] / (2 * n)) if n > 0 else 0.0
            output_equity_dict[sa_str_hhand][sa_str_vhand] = {'equity': equity, 'n': n}
        
    return output_equity_dict

def plot_preflop_equity_matrix(hand, equity_dict):
    
    hand_equity_dict = equity_dict[hand]

    matrix = np.zeros((13, 13))  # 13x13 matrix for hand equities
    n_matrix = np.zeros((13, 13))  # matrix for number of runouts tested

    for current_hand, values in hand_equity_dict.items():
        equity = values['equity']
        n = values['n']
        rank1, rank2 = current_hand[0], current_hand[1]
        suited = current_hand.endswith('s')

        r_high, r_low = RANKS.index(rank1), RANKS.index(rank2)

        if suited:
            matrix[12 - r_high, 12 - r_low] = equity  # suited hands above the diagonal
            n_matrix[12 - r_high, 12 - r_low] = n
        elif r_high == r_low:
            matrix[12 - r_high, 12 - r_low] = equity  # paired hands on the diagonal
            n_matrix[12 - r_high, 12 - r_low] = n
        else:
            matrix[12 - r_low, 12 - r_high] = equity  # Offsuit hands below the diagonal
            n_matrix[12 - r_low, 12 - r_high] = n

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the matrix with a color map
    cax = ax.imshow(matrix, cmap='coolwarm', vmin=0, vmax=1)
    cbar = fig.colorbar(cax, ax=ax, label='Equity')
    cbar.formatter = FuncFormatter(lambda x, _: f'{x * 100:.0f}%')
    cbar.update_ticks()

    # Set ticks with reversed RANKS order to align with flippedmatrix
    ax.set_xticks(np.arange(13))
    ax.set_yticks(np.arange(13))
    ax.set_xticklabels(RANKS[::-1])  
    ax.set_yticklabels(RANKS[::-1])  

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(13):
        for j in range(13):
            equity_val = matrix[i, j]
            n_val = int(n_matrix[i, j])
            if equity_val > 0:
                # Determine hand notation
                if i == j:
                    current_hand = f'{RANKS[12 - j]}{RANKS[12 - i]}'  # Paired
                elif i < j:
                    current_hand = f'{RANKS[12 - i]}{RANKS[12 - j]}s'  # Suited
                else:
                    current_hand = f'{RANKS[12 - j]}{RANKS[12 - i]}o'  # Offsuit

                ax.text(j, i, f'{current_hand}\n{equity_val * 100:.0f}%\n({n_val})',
                        ha='center', va='center', color='black', fontsize=8)
                
    total_equity = 0
    for i in range(13):
        for j in range(13):
            total_equity += matrix[i][j] * n_matrix[i][j]
    average_equity = total_equity/np.sum(n_matrix)
    
    ax.set_title(f'{hand} Preflop Equity Heatmap (Average equity = {average_equity * 100:.0f}%)')
    plt.show()

# postflop equity

def str_hand_combinations():
    combinations = []
    
    for i in range(len(RANKS)):
        for j in range(i + 1): 
            rank1 = RANKS[i]
            rank2 = RANKS[j]
        
            if rank1 == rank2:
                for suit1 in SUITS:
                    for suit2 in SUITS:
                        if suit1 > suit2:  
                            combinations.append(f"{rank1}{suit1}{rank2}{suit2}")
            else:
                for suit1 in SUITS:
                    for suit2 in SUITS:
                        combinations.append(f"{rank1}{suit1}{rank2}{suit2}")
    
    combinations.reverse()

    return combinations

def combine_equity_dicts(dict_list: list[dict]) -> dict:
    combined_dict = {}

    for raw_equity_dict in dict_list:
        for villain_hand, (doubled_wins, n) in raw_equity_dict.items():
            if villain_hand not in combined_dict:
                combined_dict[villain_hand] = {'n': 0, 'doubled_wins': 0}
            combined_dict[villain_hand]['doubled_wins'] += doubled_wins
            combined_dict[villain_hand]['n'] += n

    for villain_hand in combined_dict:
        current_dict = combined_dict[villain_hand]
        dw = current_dict['doubled_wins']
        n = current_dict['n']
        current_dict['equity'] = dw / (2 * n) if n != 0 else 0
        current_dict.pop('doubled_wins')

    return combined_dict

def postflop_equity_monte_carlo(hero_hand: str, board: str, num_iterations, thread_index=0, total_threads=1, verbose=True) -> dict:

    # Convert hero_hand and board to eval7 format
    ev7_hero_hand = ev7_hand_of_str_hand(hero_hand)
    ev7_board = ev7_board_of_str_board(board)

    # Initialize deck and remove hero hand and board cards
    deck = ev7.Deck()
    taken_ev7_card_set = set(ev7_hero_hand + ev7_board)
    deck.cards = [card for card in deck.cards if card not in taken_ev7_card_set]
    
    raw_villain_str_hands = str_hand_combinations()
    raw_villain_ev7_hands = [ev7_hand_of_str_hand(villain_hand) for villain_hand in raw_villain_str_hands]
    villain_str_and_ev7_hands = []
    for i in range(1326):
        if not any(card in raw_villain_ev7_hands[i] for card in taken_ev7_card_set):
            villain_str_and_ev7_hands.append((raw_villain_str_hands[i], raw_villain_ev7_hands[i]))

    equity_dict = {}

    for villain_hand, ev7_villain_hand in villain_str_and_ev7_hands:
        equity_dict[villain_hand] = [0, 0]  # [doubled_hero_wins, showdown_count]

    for _ in tqdm(range(num_iterations),
                  f"Thread {thread_index:>3}/{total_threads} | Postflop Monte-Carlo Simulation Progress",
                  dynamic_ncols=False,
                  mininterval=1 if total_threads != 1 else 0.1,
                  disable=not verbose):

        # Sampleremaining cards to complete the board
        ev7_remaining_board = deck.sample(5 - len(ev7_board))
        ev7_full_board = ev7_board + ev7_remaining_board

        hero_value = ev7.evaluate(ev7_hero_hand + ev7_full_board)

        for str_villain_hand, ev7_villain_hand in villain_str_and_ev7_hands:
            if any(card in ev7_remaining_board for card in ev7_villain_hand):
                continue  # Skip if villain's hand collides with the board

            villain_value = ev7.evaluate(ev7_villain_hand + ev7_full_board)

            if hero_value > villain_value:
                equity_dict[str_villain_hand][0] += 2  # Hero wins
            elif hero_value == villain_value:
                equity_dict[str_villain_hand][0] += 1  # Split pot
            # Increment showdown count
            equity_dict[str_villain_hand][1] += 1

    return equity_dict


def postflop_equity_monte_carlo_parallel(hero_hand: str, board: str, num_iterations_per_core, num_cores):
    with multiprocessing.Pool(num_cores) as pool:
        results = []

        for i in range(num_cores):
            result = pool.apply_async(postflop_equity_monte_carlo, args=(hero_hand, board, num_iterations_per_core, i, num_cores, True))
            results.append(result)

        for result in tqdm(results, "Retrieving results from all cores", total=num_cores):
            result.wait()

        equity_dicts = [result.get() for result in tqdm(results, "Combining results from all cores ")]

    combined_result = combine_equity_dicts(equity_dicts)
    return combined_result

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# RFI Quiz Methods
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# @TODO