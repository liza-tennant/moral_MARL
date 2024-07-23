from numpy.random import SeedSequence, default_rng
import os 

class my_RN_generator: 
    """ this class takes in a single master_seed (defined before each set of runs) and creates:
    - 2 streams of numbers for players 1&2, 
    - then, for each player, 9 streams of numbers to be used in the experiments.
    
    It allows us to generate 10 random numbers for every player: 
    [0] 'empty_Q': random move when dilemma memory buffer is empty, 
    [1] 'eps_prob': probability to compare against eps for dilemma move, 
    [2] 'eps_move': random dilemma move due to eps, 
    [3] 'static_random': move for a static agent with strategy=='random', 
    [4] 'random_state': initial random state to begin learning,
    [5] 'DQN_weights': seed for dilemma Q-network weights initialisation (INT),
    [6] 'memory_sampling': seed for sampling dilemma experience from memory buffer (INT) 
    [7] 'selection_empty_Q': random selection when selection memory buffer is empty,
    [8] 'selection_eps_prob': probability to compare against eps for selection move,
    [9] 'selection_eps_move':  random selection due to eps
    [10]'sel_DQN_weights': seed for selection Q-network weights initialisation (INT),
    [11]'sel_memory_sampling': seed for sampling selection experience from memory buffer (INT) 

    Additionaly, we generate the following 3 streams for the game environment itself: 
    [0] 'sample_leading_player': random sampling of leading_player (player1) from the populaiton as we loop over iterations 
    [1] 'sample_opponent': randomly sample opponent (player2) in game.random_matching (if not using partner selection)
    [2] 'initial env_state': random initial state for the game environment
    [3] 'shuffle_pairs': shuffle pairs for parallel dilemma games 
    """

    def __init__(self, master_seed, population_size=None):
        self.master_seed = master_seed
        if population_size: 
            self.population_size = population_size
            self.player_streams = {}
            self.game_streams= {}
        else: #in dyadic case 
            self.player1_streams = None 
            self.player2_streams = None 


    def generate(self, destination_folder):
        ss = SeedSequence(self.master_seed)

        if self.population_size: #if poulation case
            child_seeds = ss.spawn(self.population_size+1)
            #save child seeds to file for future reference
            if not os.path.isdir(str(destination_folder)):
                os.makedirs(str(destination_folder))

            #save child seeds
            with open(f'{destination_folder}/child_seeds.txt', 'w') as fp:
                for item in child_seeds:
                    fp.write("%s\n" % str(item)) # write each item on a new line
                print('saved child_seeds for all players + the population')

            #generate our streams of numbers for each of the two players
            for i in range(self.population_size): 
                grandchildren_player_i = child_seeds[i].spawn(12)
                #player_i_streams = [default_rng(s) for s in grandchildren_player_i]
                player_i_streams = dict(zip(range(12), [default_rng(s) for s in grandchildren_player_i]))

                self.player_streams[f'player{i}'] = player_i_streams

            grandchildren_game = child_seeds[-1].spawn(4)
            self.game_streams = dict(zip(range(4), [default_rng(s) for s in grandchildren_game]))

        else: #in dyadic case 
            # Spawn off 2 child SeedSequences (2 players) to pass to grandchild/child processes.
            child_seeds = ss.spawn(2) 

            #save child seeds to file for future reference
            if not os.path.isdir(str(destination_folder)):
                os.makedirs(str(destination_folder))
            
            #save child seeds
            with open(f'{destination_folder}/child_seeds.txt', 'w') as fp:
                for item in child_seeds:
                    fp.write("%s\n" % str(item)) # write each item on a new line
                print('saved child_seeds for all players in the population')

            #generate our streams of numbers for each of the two players
            grandchildren_player1 = child_seeds[0].spawn(10)
            player1_streams = {s:default_rng(s) for s in grandchildren_player1}
            self.player1_streams = player1_streams

            grandchildren_player2 = child_seeds[1].spawn(10)
            player2_streams = {s:default_rng(s) for s in grandchildren_player2}
            self.player2_streams = player2_streams



