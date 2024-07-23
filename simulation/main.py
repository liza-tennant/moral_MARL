#!/cs/academic/phd3/lkarmann/my_env/bin/python3.8
# if running on a cluster
import pandas as pd
import numpy as np 
#from matplotlib.pyplot import figure
import os
import argparse

#from test import Test #test out importing a module in this environment
#test_object = Test()
#test_object.test_function()

#import own modules that contain the player, environment and random nummber generator. 
from environment import Game_for_population_learning
from player import Player
from rn_generator import my_RN_generator

#from qnetwork import QNetwork
#from buffer import ReplayBuffer
from config import * 

#import torch
#import torch.nn.functional as F
#import torch.optim as optim

from itertools import chain
#import random 
import pickle 

import torch 

from utils.time_profiler import TimeProfiler 


#os.chdir("~/Library/Mobile\ Documents/com~apple~CloudDocs/PhD_data")

#set via config.py: PAYOFFMAT_IPD, my_game, study_type

if study_type == 'population_withselection':


    def DQN_population_withepisodes_withselection(game, global_history, num_iter, RNG, method, population_config): 
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(f"Using {device} device")

        population_size = population_config['population_size']
        #selection_size = population_config['selection_size']
        #state_size = population_config['state_size']

        tp.start("    set initial_selection_states")
        #initialise different random selection_states for every player (randomly)
        initial_selection_states = []
        #TO DO make this a player attribute ? 
        for idx in range(population_size): #loop over the population in order  
            if state_pairs:
                initial_selection_state_playeri = [RNG.player_streams[f'player{i}'][4].choice([(0,0), (0,1), (1,0), (1,1)]) for i in range(population_size-1)] #choice(..., 1)
                initial_selection_state_playeri = [tuple(state) for state in initial_selection_state_playeri]
                initial_selection_states.append(tuple(initial_selection_state_playeri)) #append in order 
            else: #if state not pairs 
                initial_selection_state_playeri = [RNG.player_streams[f'player{i}'][4].choice([0, 1]) for i in range(population_size-1)] #choice(..., 1)
                #initial_selection_state_playeri = tuple(initial_selection_state_playeri)
                initial_selection_states.append(initial_selection_state_playeri) #append in order
        tp.end("    set initial_selection_states")

        tp.start("    set initial game.env_state")
        #initialise random environment state (randomly) --> used to defer dilemma states for every player
        #set game.env_state randomly to begin with 
        if state_pairs: 
            game.env_state = tuple([RNG.game_streams[2].choice((0,0), (0,1), (1,0), (1,1)) for i in range(population_size)]) #initialise the environment state to a random state
        else: #if state not using pairs 
            game.env_state = [RNG.game_streams[2].choice([0,1]) for i in range(population_size)] #initialise the environment state to a random state
        tp.end("    set initial game.env_state")


        #also initalise Q-values storage and loss
        if record_Qvalues == 'both':
            history_Qvalues_dilemma_local = {}
            history_Qvalues_selection_local = {}  
            history_Qvalues_dilemma_local['format'] = '{state, action, iteration: Q-value dilemma}'
            history_Qvalues_selection_local['format'] = '{state, action, iteration: Q-value selection}'
        #if record_Qvalues = False, do not create a placeholder for Qvalue storage 
        if record_Loss == 'both': 
            history_loss_dilemma = {}
            history_loss_selection = {}
            #NOTE TO DO 
        elif record_Loss in ['dilemma', 'selection']:
            history_loss = {}
        else: 
            history_loss = {}

        tp.start("    loop over game.population and initialise possible_opponents, Q-values and loss")
        #initalise each player's possible_opponents, Q-values and loss
        for player in game.population: 
            p_idx = game.population.index(player)
            player.current_selection_state = initial_selection_states[p_idx]
            if False: #testing with given initial state - assign this manually 
                if dilemma_state_uses_identity:
                    player.current_state = ((0, 0), 0)
                    player.current_state = list(chain(*(i if isinstance(i, tuple) else (i,) for i in player.current_state)))
                else: 
                    if state_pairs:
                        player.current_state = (0, 0)
                        player.current_selection_state = tuple([(0, 0) for i in range(population_size-1)])
                    else: #if state not using pairs 
                        player.current_state = 0 
                        player.current_selection_state = [0 for i in range(population_size-1)]

            #within the player object, record possible_opponent_indices local and global, and mapping from one to the other 
            player.possible_opponent_indices = list(range(population_size-1)) #store possible local opponent indices for this player
            player.possible_opponent_indices_global = list(range(population_size))
            player.possible_opponent_indices_global.remove(game.population.index(player)) #store possible global opponent indices for this player

            player.index_mapping_localtoglobal = dict(zip( 
                player.possible_opponent_indices, #local indices 
                player.possible_opponent_indices_global #global indices 
                ))
            player.index_mapping_globaltolocal = dict(zip( 
                player.possible_opponent_indices_global, #global indices 
                player.possible_opponent_indices, #local indices
                ))

            #set up other player atributes & history objects for Qvalues and Loss 
            if 'Q-Learning' in player.strategy:
            #if player.strategy not in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec']:
                player.alg_type = method
                player.current_loss = 0
                
                if False: 
                    player.loss_history = np.full((num_iter*population_size), np.nan) #MAYBE TO DO fix loss storage to be more efficient?
                    player.selection_loss_history = np.full((num_iter*population_size), np.nan) 
                player.running_selection_loss = 0
                player.running_loss = 0

                if record_Loss == 'both':
                    history_loss_dilemma[f'player{p_idx}'] = []
                    history_loss_selection[f'player{p_idx}'] = []
                elif record_Loss in ['dilemma', 'selection']: #else: 
                    history_loss[f'player{p_idx}'] = []
                
                if player.record_Qvalues == 'both':
                    #initialise each player's dict#TO DO check if this is needed 
                    history_Qvalues_dilemma_local[f'player{p_idx}'] = {}  
                    history_Qvalues_selection_local[f'player{p_idx}'] = {} 

                #if dilemma_state_uses_identity:
                #    player.random_dilemma_identity = RNG.player_streams[f'player{p_idx}'][4].choice(range(population_size-1)) #player.possible_opponent_indices #randomly choose an opponent's identity
  #NOTE TO DO understand if the identity is using each player's local observation of their opponents? (i.e. of size population_size-1)

        tp.end("    loop over game.population and initialise possible_opponents, Q-values and loss")

        #running_loss_player1 = 0 #TO DO 
        #running_loss_player2 = 0 #TO DO 
        #running_selection_loss_player1 = 0

        #if episodes== False: 
        #    population_temp = list(range(population_size)) #for looping over population without replacement in random order 

        tp.start("    episode loop")
        #iteration = -1 
        iteration_sel = -1 
        iteration_dil = -1
        for episode in range(num_iter): #e.g. num_iter - e.g. encounters
            tp.start("    one episode")

            #### PARTNER SELECTION phase ####
            game.selection_pairs = [] #reset selection_pairs attribute 

            tp.start("        selection phase - simultaneously for all players")
            for player in game.population: 
                iteration_sel += 1
                p1_idx = game.population.index(player)

                if episode > 0: #on the first iteration, use the random initial_selection_state
                    #update player.current_selection_state using latest observations form the environment 
                    #player1 = game.population[p1_idx]
                    player.current_selection_state = game.env_state.copy()
                    mymove = player.current_selection_state.pop(p1_idx) #remove self from current_selection_state
                    #the above will modify player1.current_selection_state directly, but not game.env_state

                if random_matching: 
                    tp.start("        game.random_matching")
                    #### RANDOMLY MATCHED PARTNERS ####
                    p2_idx = game.random_matching(p1_idx, RNG)
                    tp.end("        game.random_matching")

                else: #use parter_selection  
                    #### PARTNER SELECTION ####
                    #we can loop over populaiton in order since the selections are based on the same game.env_state, so conceptually speaking they happen simultaneously                    
                    tp.start("        game.partner_selection")
                    p2_idx = game.partner_selection(p1_idx, iteration_sel, num_iter, global_history, RNG) #note p1_idx, p2_idx are GLOBAL indices 
                    tp.end("        game.partner_selection")

                game.selection_pairs.append(tuple([p1_idx, p2_idx]))
                player.t_step_selection = (player.t_step_selection + 1) #NOTE TO DO maybe move this to the previous selection loop? 


            #### DILEMMA phase - all pairs####
            RNG.game_streams[3].shuffle(game.selection_pairs)
            #loop over the selection_pairs in a random order, so that if a player has multiple dilemma trajectories, there are no order effects 
            for p1_idx, p2_idx in game.selection_pairs:
                iteration_dil += 1 
            
                tp.start("        update px_idx_by_py, player.current_state")
                player1 = game.population[p1_idx]
                player2 = game.population[p2_idx]

                p2_idx_by_p1 = player1.current_selection_idx
                p1_idx_by_p2 = player2.index_mapping_globaltolocal[p1_idx] 

                #get dilemma state for each player
                if dilemma_state_uses_identity:
                    #if player1.t_step_dilemma == 0:
                    #    player1.current_state = (game.env_state.copy()[p2_idx], player1.random_dilemma_identity)
                    #if player2.t_step_dilemma == 0:
                    #    player2.current_state = (game.env_state.copy()[p1_idx], player2.random_dilemma_identity)
                    #else: 
                    #update player1.current_state to involve the partner they are now facing
                    player1.current_state = (game.env_state.copy()[p2_idx], p2_idx_by_p1)
                    player2.current_state = (game.env_state.copy()[p1_idx], p1_idx_by_p2)
                    #NOTE The above modifies the original objects in initial_states too !!!! ??

                else: #if not using identity in dilemma state 
                    player1.current_state = game.env_state.copy()[p2_idx]
                    player2.current_state = game.env_state.copy()[p1_idx]
                tp.end("        update px_idx_by_py, player.current_state")


                #### pair PLAY THE DILEMMA ####
                tp.start("        game.step_simpified")
                #execute a step that interacts with the environment & updates global_history behind the scenes 
                action_player1, action_player2, next_state_player1, next_state_player2, reward_learning_player1, reward_learning_player2 = game.step_simplified(
                    player1, player2, p1_idx, p2_idx, iteration_dil, global_history, num_iter, RNG) #p2_idx_by_p1, p1_idx_by_p2, 
                tp.end("        game.step_simpified")

                #for debugging
                #print(f'action_player1 (p_idx {p1_idx}), action_player2 (p_idx {p2_idx}): ', action_player1, action_player2)

                player1.t_step_dilemma = (player1.t_step_dilemma + 1) 
                player2.t_step_dilemma = (player2.t_step_dilemma + 1) 
                #NOTE understand if the above is being used correctly? 

                #record dilemma reward as selection reward - for use in network updates later 
                player1.current_reward_selection = reward_learning_player1

                #overwrite own latest_move attributes for each player with whichever game was played last 
                player1.latest_move = action_player1
                player2.latest_move = action_player2

                if 'Q-Learning' in player1.strategy: 
                    tp.start("        player1.dilemma_memory.add") # store the dilemma trajectory 
                    player1.dilemma_memory.add(
                        player1.current_state, #NOTE if using identity - this uses the numeric local/CUSTOM identity idx
                        action_player1, 
                        reward_learning_player1, 
                        next_state_player1) #record next state based on interaction with this same opponent 
                    tp.end("        player1.dilemma_memory.add")


                if player2_learns and (player2.strategy not in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec', 'GilbertElliot_typed', 'GilbertElliot_typee']):
                    tp.start("        player2.dilemma_memory.add")
                    player2.dilemma_memory.add(
                        player2.current_state, 
                        action_player2, 
                        reward_learning_player2, 
                        next_state_player2)
                    tp.end("        player2.dilemma_memory.add")

                #### STORE RNs & selection attributes ####
                if record_RNs: 
                    print('TO DO for episodes !!!!')
                    #TO DO fix this for episodes 
                    tp.start("        record RNs in global_history")
                    global_history.loc[iteration_dil, [#'game_RN3', #NOTE TO DO add game_RN3 storage fter we fix setting it above 
                                                'p1_RN0', 'p1_RN1', 'p1_RN2', 'p1_RN4',
                                                'p1_RN5', 'p1_RN6',
                                                'p1_RN10', 'p1_RN11',
                                                'p1_RN7', 'p1_RN8', 'p1_RN9']] = [
                                                    #game.RNs['shuffle_pairs'], 
                                                    player1.RNs['empty_Q'], player1.RNs['eps_prob'], player1.RNs['eps_move'], player1.RNs['random_state'],
                                                    player1.RNs['dil_DQN_weights'], player1.RNs['dil_memory_sampling'],
                                                    player1.RNs['sel_DQN_weights'], player1.RNs['sel_memory_sampling'],
                                                    player1.RNs['selection_empty_Q'], player1.RNs['selection_eps_prob'], player1.RNs['selection_eps_move']
                                                    ]
                    global_history.loc[iteration_dil, ['p2_RN0', 'p2_RN1', 'p2_RN2', 'p2_RN4',
                                                'p2_RN5', 'p2_RN6',
                                                'p2_RN10', 'p2_RN11'#,
                                                #'p2_RN7', 'p2_RN8', 'p2_RN9'
                                                ]] = [
                                                        player2.RNs['empty_Q'], player2.RNs['eps_prob'], player2.RNs['eps_move'], player2.RNs['random_state'],
                                                        player2.RNs['dil_DQN_weights'], player2.RNs['dil_memory_sampling'],
                                                        player2.RNs['sel_DQN_weights'], player2.RNs['sel_memory_sampling']#,
                                                        #player2.RNs['selection_empty_Q'], player2.RNs['selection_eps_prob'], player2.RNs['selection_eps_move']
                                                        ]
                    tp.end("        record RNs in global_history")

                global_history.loc[iteration_dil, ['episode']] = [episode]
                if record_reason: 
                #    #idx = game.selection_pairs.index((p1_idx, p2_idx))
                    global_history.loc[iteration_dil, ['reason_selection_player1']] = [
                        player1.reason_selection]    #reasons_selection[idx]]
                    
                    player1.reason_selection = None #reset attribute for next episode

                if player1.strategy not in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec', 'GilbertElliot_typed', 'GilbertElliot_typee']:
                    tp.start("        record player1 selection state etc in global_history")

                    #add selection data to global_history: 
                    if record_eps: 
                        #print('TO DO for eps_selection (see env.partner_selection) !!!!')
                        global_history.loc[iteration_dil, ['eps_selection_player1']] = [
                            player1.eps_selection]
                        player1.eps_selection = None #reset attribute for next episode
                    if record_selection_state:
                        global_history.loc[iteration_dil, ['selection_state_player1']] = [
                            str(player1.current_selection_state)]
                        global_history.loc[iteration_dil, ['env_state']] = [
                            str(game.env_state)]

                    global_history.loc[iteration_dil, ['selection_player1']] = [
                        player1.current_selection_idx] #note this records the LOCAL index used by player1

                    if state_pairs: 
                        print('TO DO for state_pairs !!!!')
                        if False: 
                            opponent_moves_available = [pair[0] for pair in player1.current_selection_state]
                            global_history.loc[iteration_dil, ['cooperative_sel_available']] = [
                                0 in opponent_moves_available]
                            global_history.loc[iteration_dil, ['selected_prev_move']] = [
                                player1.current_selection_state[player1.current_selection_idx][0]]
                            
                            #update player1's selection state -  for the leading player's observation of their current opponent
                            temp1 = list(player1.current_selection_state.copy())
                            temp1[p2_idx_by_p1] = (action_player2, action_player1) #insert the opponent's move as a response to leading player's move  
                            selection_next_state_player1 = temp1 #this will only be used in the selection_memory storage, not actually on the next step of the selectio nfor this agent   
                    else: #if state not using state_pairs
                        opponent_moves_available = [move for move in player1.current_selection_state]
                        global_history.loc[iteration_dil, ['cooperative_sel_available']] = [
                            0 in opponent_moves_available]

                        global_history.loc[iteration_dil, ['selected_prev_move']] = [
                            player1.current_selection_state[player1.current_selection_idx]]
                        
                    tp.end("        record player1 selection state etc in global_history")
            
            #### UPDATE env_state, using each player's latest move ####
            for player in game.population: 
                p_idx = game.population.index(player)
                game.env_state[p_idx] = player.latest_move
                # this gets carried over to the next episode, and gets used to compute next_selection_state for each player 

            #### UPDATE networks and store loss etc. ####
            for player in game.population: 
                p_idx = game.population.index(player)

                if 'Q-Learning' in player.strategy: #not in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec']:
                    if state_pairs: 
                        print('TO DO for state_pairs !!!!')
                        if False:   
                                #update game.env_state, using player2's global index 
                                game.env_state[p1_idx] = (action_player1, action_player2) #insert the leading player's move as a response to the opponent's move
                                game.env_state[p2_idx] = (action_player2, action_player1) #insert the leading player's move as a response to the opponent's move
                    else: 
                        #### UPDATE each player's next_selection_state ####
                        player.next_selection_state = game.env_state.copy()
                        mymove = player.next_selection_state.pop(p_idx) 

                        #### RECORD selection IN MEMORY ####
                        tp.start("        player.selection_memory.add")
                        #store experience for learning to select partner 
                        player.selection_memory.add(
                            player.current_selection_state,
                            player.current_selection_idx, #this was updated in gamme.partner_selection
                            player.current_reward_selection, #reward_learning_player1, #bring forward the reward from the dilemma and repeat it in the selection trajecotory 
                            player.next_selection_state
                            )
                        tp.end("        player.selection_memory.add")

                    #### UPDATE NETWORKS for each player ####
                    tp.start("        update networks for all players in a loop")
                    #### SELECTION network 
                    if len(player.selection_memory) >= 1:
                        tp.start(f"            player{p_idx}.selection_memory.sample()")
                        selection_states, selections, selection_rewards, selection_next_states = player.selection_memory.sample() 
                        tp.end(f"            player{p_idx}.selection_memory.sample()")

                        #for debugging
                        #print(f'selection rewards p_idx {p_idx}: ', selection_rewards)

                        tp.start(f"            player{p_idx}.update_selection_Qnetwork")
                        player.update_selection_Qnetwork(selection_states, selections, selection_rewards, selection_next_states, gamma) #NB this updates the player's QValues_expected and Qvalues_targets attributes
                        tp.end(f"            player{p_idx}.update_selection_Qnetwork")

                        tp.start(f"            update player{p_idx}.selection_loss_history & print running loss")
                        selection_loss = player.current_selection_loss 
                        if selection_loss >= 0.0001:
                            selection_loss = round(selection_loss.item(), 4)
                            player.selection_loss_history[iteration_sel] = selection_loss
                        elif selection_loss: 
                            player.selection_loss_history[iteration_sel] = 0 #instead of '<0.0001'
                        else: 
                            player.selection_loss_history[iteration_sel] = np.nan

                        player.selection_memory.refresh(BUFFER_SIZE) #just in case 

  #NOTE the below will rely on latest iteration_dil available during this episode - and will not loop over all of them 
                        #print statistics - running average 
                        player.running_selection_loss += selection_loss
                        if iteration_sel % population_size == 1: #iteration_dil+1 == BATCH_SIZE: #print first available loss 
                            print(f'[episode {episode}, population size {population_size}, sel {iteration_sel + 1:5d}] selection loss for player {p_idx}: {player.running_selection_loss:.4f}')
                        if iteration_sel % 1000 == 999:    # print every 1000 mini-batches
                            print(f'[episode {episode}, population size {population_size}, sel {iteration_sel + 1:5d}] selection loss for player {p_idx}: {player.running_selection_loss / 1000:.4f}')
                            player.running_selection_loss = 0.0
                        tp.end(f"            update player{p_idx}.selection_loss_history & print running loss")
                    else:                         player.selection_loss_history[iteration_sel] = np.nan

                    ##### DILEMMA network 
                    if len(player.dilemma_memory) >= 1: 
                        #NOTE we now sample all epxeriences from the dilemma_memory in a minibatch - for more stability  
                        tp.start(f"            player{p_idx}.dilemma_memory.sample()")
                        states, actions, rewards, next_states = player.dilemma_memory.sample() #this will sample in order of entry - oldest experience first
                        tp.end(f"            player{p_idx}.dilemma_memory.sample()")
                        ### apply training loop & save loss
                        tp.start(f"            player{p_idx}.update_Qnetwork")
                        player.update_Qnetwork(states, actions, rewards, next_states, gamma) #this updates the player's QValues_expected and Qvalues_targets attributes
                        #TO DO understand if the above is taking in the right sate format 
                        tp.end(f"            player{p_idx}.update_Qnetwork")

                        tp.start(f"            update player{p_idx}.loss_history & print running loss")
                        loss_ = player.current_loss # store only one dilemma_loss per episode = average MSE for all dilemma trajectories 
                        if loss_ >= 0.0001:
                            loss_ = round(loss_.item(), 4)
                            player.loss_history[iteration_dil] = loss_

                        elif loss_: 
                            player.loss_history[iteration_dil] = 0 #'<0.0001'
                        else: 
                            player.loss_history[iteration_dil] = np.nan
                        
                        #print statistics
                        player.running_loss += loss_
                        #TO DO fix running_loss 
                        if iteration_dil  == 0: #print first available loss 
                            print(f'[{iteration_dil + 1:5d}] dilemma loss for player {p_idx}: {player.running_loss:.4f}')
                        #if iteration % 2000 == 1999:    # print every 2000 mini-batches
                        if iteration_dil % 1000 == 999:    # print every 2000 mini-batches
                            print(f'[{iteration_dil + 1:5d}] dilemma loss for player {p_idx}: {player.running_loss / 1000:.4f}')
                            player.running_loss = 0.0
                        tp.end(f"            update player{p_idx}.loss_history & print running loss")

                        player.dilemma_memory.refresh(BUFFER_SIZE) #just in case

                    else: 
                        player.loss_history[iteration_dil] = np.nan

                    tp.end("        update networks for all players in a loop")

                    #for debugging 
                    # print(f'current player{p_idx}.Q_values_dilemma: {player.Qvalues_expected_dilemma}')

                    if False: #### RECORD Qvalues & Loss ####
                        tp.start(f"    store player.{record_Loss}_loss_history in history_loss & player.Qvalues in history-Qvalues")
                        if record_Loss == 'both':
                            history_loss_dilemma[f'player{p_idx}'] = player.loss_history #.append(player.loss_history)
                            history_loss_selection[f'player{p_idx}'] = player.selection_loss_history #.append(player.selection_loss_history)
                        elif record_Loss == 'selection':
                            history_loss[f'player{p_idx}'] = player.selection_loss_history #.append(player.selection_loss_history)
                        elif record_Loss == 'dilemma':
                            history_loss[f'player{p_idx}'] = player.loss_history #.append(player.loss_history)

    #QUESTION: should the below be happening after all episode, or once every episdoe 
                        if record_Qvalues == 'both':
                            history_Qvalues_dilemma_local[f'player{p_idx}'] = player.Qvalues_expected_dilemma.copy()#.iloc[:, 0:selection_size] 
                            history_Qvalues_selection_local[f'player{p_idx}'] = player.Qvalues_expected_selection.copy()#.iloc[:, 0:selection_size] 
                            #history_Qvalues_dilemma_local[f'player{p1_idx}'][int(iteration/record_Qvalues_every)] = player1.Qvalues_expected_dilemma.copy()#.iloc[:, 0:selection_size] 
                            #history_Qvalues_selection_local[f'player{p1_idx}'][int(iteration/record_Qvalues_every)] = player1.Qvalues_expected_selection.copy()#.iloc[:, 0:action_size] 
                        tp.end(f"    store player.{record_Loss}_loss_history in history_loss & player.Qvalues in history-Qvalues")

            tp.end("    one episode")
        tp.end("    episode loop")

        #### RECORD Qvalues & Loss ####
        for player in game.population: 
            if 'Q-Learning' in player.strategy: #not in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec']:
                p_idx = game.population.index(player)

                tp.start(f"    store player.{record_Loss}_loss_history in history_loss & player.Qvalues in history-Qvalues")
                if record_Loss == 'both':
                    history_loss_dilemma[f'player{p_idx}'] = player.loss_history #.append(player.loss_history)
                    history_loss_selection[f'player{p_idx}'] = player.selection_loss_history #.append(player.selection_loss_history)
                elif record_Loss == 'selection':
                    history_loss[f'player{p_idx}'] = player.selection_loss_history #.append(player.selection_loss_history)
                elif record_Loss == 'dilemma':
                    history_loss[f'player{p_idx}'] = player.loss_history #.append(player.loss_history)

                if record_Qvalues == 'both':
                    history_Qvalues_dilemma_local[f'player{p_idx}'] = player.Qvalues_expected_dilemma.copy()#.iloc[:, 0:selection_size] 
                    history_Qvalues_selection_local[f'player{p_idx}'] = player.Qvalues_expected_selection.copy()#.iloc[:, 0:selection_size] 
                    #history_Qvalues_dilemma_local[f'player{p1_idx}'][int(iteration/record_Qvalues_every)] = player1.Qvalues_expected_dilemma.copy()#.iloc[:, 0:selection_size] 
                    #history_Qvalues_selection_local[f'player{p1_idx}'][int(iteration/record_Qvalues_every)] = player1.Qvalues_expected_selection.copy()#.iloc[:, 0:action_size] 
                tp.end(f"    store player.{record_Loss}_loss_history in history_loss & player.Qvalues in history-Qvalues")



    
        #if player1.type not in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random']:
        ## NOTE the below would now be saved at every iteration --> inefficient ! 
        #    summary = player1.qnetwork_local.load_state_dict
        #    store_network_architecture(counter, destination_folder, summary)
        
        #to save the trained model 
        #PATH = './player1_IPD_net.pth' #saving for one example run 
        #torch.save(player1.qnetwork_local.state_dict(), PATH)

        empty = None #initialise placeholder for 'storing' empty Loss or Qvalue object

        # calculate social outcome metrics from each pair of rewards observed 
        tp.start("    calculate social outcomes")
        global_history = calculate_social_outcomes(global_history) 
        tp.end("    calculate social outcomes")


        if record_Qvalues == 'both' and record_Loss == 'both':
            return global_history, history_loss_dilemma, history_loss_selection, history_Qvalues_dilemma_local, history_Qvalues_selection_local
        elif record_Qvalues == 'both' and record_Loss in ['dilemma', 'selection']: #if recoding only one of the losses 
            return global_history, history_loss, history_Qvalues_dilemma_local, history_Qvalues_selection_local
        elif record_Qvalues == 'both': #if not recording any Loss 
                return global_history, empty, history_Qvalues_dilemma_local, history_Qvalues_selection_local
        
        elif record_Loss == 'both': #if not stoing Qvalues but sotring Loss 
            return global_history, history_loss_dilemma, history_loss_selection, empty, empty
        elif record_Loss in ['dilemma', 'selection']: #if recoding only one of the losses 
            return global_history, history_loss, empty, empty
        else: #if not storing Qvalues or Loss  
            return global_history, empty, empty  #, result

    def calculate_social_outcomes(global_history):
        pay1, pay2 = global_history['reward_game_player1'], global_history['reward_game_player2']

        global_history['reward_collective'] = pay1 + pay2
        global_history['reward_gini'] = 1 - ((abs(pay1 - pay2)) / (pay1 + pay2))
        global_history['reward_min'] = global_history[['reward_game_player1', 'reward_game_player2']].min(axis=1)

        return global_history

    def run_one_population_episode_with_selection(run, population, num_iterations, my_RNG, destination_folder, method, population_config):
            #define payoff matrix from config file and my_game 
            if my_game == 'IPD':
                PAYOFFMAT = PAYOFFMAT_IPD
            elif my_game == 'VOLUNTEER':
                PAYOFFMAT = PAYOFFMAT_VOLUNTEER
            elif my_game == 'STAGHUNT':
                PAYOFFMAT = PAYOFFMAT_STAGHUNT

            tp.start("create Game_for_population_learning")
            game = Game_for_population_learning(population, PAYOFFMAT) #uses a population-specific step function - TO DO write Game_for_population_learning as super() class of Game_for_learning
            #possible_actions = [[0, 1], [0, 1]] #shape = state set [C,D]; action set [C,D]
            tp.end("create Game_for_population_learning")

            empty = None #initialise placeholder for 'storing' empty Loss or Qvalue object

            tp.start("create empty global_history df")
            global_history = pd.DataFrame.from_dict({'episode':[None], 'title_p1':[None], 'title_p2':[None], 'idx_p1':[None], 'idx_p2':[None], 
                                                    #'selection_state_player1':[None], 'selection_state_player2':[None],
                                                    'selection_player1':[None], 
                                                    #'selection_next_state_player1':[None], 'selection_next_state_player2':[None],
                                                    'cooperative_sel_available':[None], 'selected_prev_move':[None],
                                                    'action_player1':[None], 'action_player2':[None], 
                                                    #'eps_selection_player1':[None], 'reason_selection_player1':[None],
                                                    'state_player1':[None], 'state_player2':[None], 
                                                    'reward_game_player1':[None], 'reward_game_player2':[None], 
                                                    #'next_state_player1':[None], 'next_state_player2':[None], 
                                                    'reward_intrinsic_player1':[None], 'reward_intrinsic_player2':[None], 
                                                    #'reward_collective':[None], 'reward_gini':[None], 'reward_min':[None],
                                                    'reward_learning_player1':[None], 'reward_learning_player2':[None],
                                                    #'eps_player1':[None], 'eps_player2':[None], 
                                                    #'reason_player1':[None], 'reason_player2':[None],
                                                    #'RNs_player1':[None], 'RNs_player2':[None]
                                                    }) 
            if record_selection_state == True:
                global_history['selection_state_player1'] = [None]
                #global_history['selection_state_player2'] = [None]
                if episodes == False: 
                    global_history['selection_next_state_player1'] = [None]
                #global_history['selection_next_state_player2'] = [None]
                global_history['env_state'] = [None]
            
            if record_dilemma_nextstate == True: 
                global_history['next_state_player1'] = [None]
                global_history['next_state_player2'] = [None]

            if record_eps == True: 
                global_history['eps_player1'] = [None]
                global_history['eps_player2'] = [None]
                global_history['eps_selection_player1'] = [None] 
            
            if record_reason == True: 
                global_history['reason_player1'] = [None]
                global_history['reason_player2'] = [None] 
                global_history['reason_selection_player1'] = [None] 

            if record_RNs == True:
                for colname in ['p1_RN0','p1_RN1','p1_RN2','p1_RN3','p1_RN4','p1_RN5','p1_RN6','p1_RN7','p1_RN8','p1_RN9','p1_RN10','p1_RN11',
                                'p2_RN0','p2_RN1','p2_RN2','p2_RN3','p2_RN4','p2_RN5','p2_RN6','p2_RN7','p2_RN8','p2_RN9','p2_RN10','p2_RN11',
                                'game_RN0','game_RN1']:
                    global_history[colname] = [None] 
            tp.end("create empty global_history df")

            tp.start("DQN_population_withepisodes_withselection")
            if method == 'DQN': 
                #TO DO maybe return , history_Qvalues_local_player1, history_Qvalues_target_player1, history_loss_player1, history_Qvalues_local_player2, history_Qvalues_target_player2, history_loss_player2 here 
                if episodes == True: 
                    if record_Qvalues == 'both':
                        if record_Loss == 'both':
                            global_history, history_loss_dilemma, history_Loss_selection, history_Qvalues_dilemma_local, history_Qvalues_selection_local = DQN_population_withepisodes_withselection(game, global_history, num_iterations, my_RNG, method, population_config)
                            return global_history, history_loss_dilemma, history_Loss_selection, history_Qvalues_dilemma_local, history_Qvalues_selection_local
                        elif record_Loss in ['dilemma', 'selection']:
                            global_history, history_loss, history_Qvalues_dilemma_local, history_Qvalues_selection_local = DQN_population_withepisodes_withselection(game, global_history, num_iterations, my_RNG, method, population_config)
                            return global_history, history_loss, history_Qvalues_dilemma_local, history_Qvalues_selection_local
                        else: #if not recording Loss 
                            global_history, empty, history_Qvalues_dilemma_local, history_Qvalues_selection_local = DQN_population_withepisodes_withselection(game, global_history, num_iterations, my_RNG, method, population_config)
                            return global_history, empty, history_Qvalues_dilemma_local, history_Qvalues_selection_local
                    
                    elif record_Loss == 'both': #not recording Qvalues but recording Loss
                        global_history, history_loss_dilemma, history_Loss_selection, empty, empty = DQN_population_withepisodes_withselection(game, global_history, num_iterations, my_RNG, method, population_config)
                        return global_history, history_loss_dilemma, history_Loss_selection, empty, empty
                    elif record_Loss in ['dilemma', 'selection']:
                        global_history, history_loss, empty, empty = DQN_population_withepisodes_withselection(game, global_history, num_iterations, my_RNG, method, population_config)
                        return global_history, history_loss, empty, empty
                    
                    else: #if not recording Qvalues either 
                        global_history, empty, empty = DQN_population_withepisodes_withselection(game, global_history, num_iterations, my_RNG, method, population_config)
                        return global_history, empty, empty 
                    
                else: #if not using episodes 
                #NOTE TO DO fix loss storage here if not using episodes 
                    if record_Qvalues == 'both':
                        global_history, history_loss, history_Qvalues_dilemma_local, history_Qvalues_selection_local = DQN_population_withselection(game, global_history, run, num_iterations, my_RNG, destination_folder, method, population_config)
                        return global_history, history_loss, history_Qvalues_dilemma_local, history_Qvalues_selection_local
                    else:
                        global_history, history_loss, history_Qvalues_local = DQN_population_withselection(game, global_history, run, num_iterations, my_RNG, destination_folder, method, population_config)
                        return global_history, history_loss, None
                #DQN_population_2learners_withselection(game, global_history, run, num_iterations, my_RNG, destination_folder)
            tp.end("DQN_population_withepisodes_withselection")

    def run_population_and_save_withselection(num_runs, num_iterations, destination_folder, titles, method, population_config):
        '''function to run num_runs encounters between a population of players,
        and save global_history, Q-values and Loss. '''
        if episodes: 
            iterations = 'episodes'
        else:
            iterations = 'iterations'
        print(f'running population {titles}, {num_runs} runs, {num_iterations} {iterations} each, storing in {destination_folder}')
        #for title in titles: 
        #    if 'QL' not in title:
        #        return '!! this is not the right function for these player types !!'

        population_size = population_config['population_size']


        if not os.path.isdir('results/'+ str(destination_folder)):
            os.makedirs('results/'+ str(destination_folder))
        destination_folder = 'results/'+ str(destination_folder)

        tp.start("store_population_learning_parameters")
        store_population_learning_parameters(destination_folder, method, titles, population_config)
        tp.end("store_population_learning_parameters")

        #RESULTS_list = list()
        #TO DO store greedy policy as well in RESULTS 
        
        tp.start("create my_RNG & .generate")
        #instantiate the RN_generator before I run my n runs - so that all n runs share a single set of RN streams (4, to be exact) and read from it sequentially
        my_RNG = my_RN_generator(master_seed=master_seed, population_size=population_size) #parameter set via config.py 
        my_RNG.generate(destination_folder=destination_folder) #generate all RN streams for all players, store these in my_RNG.player_streams 
        tp.end("create my_RNG & .generate")

        tp.start("all runs in num_runs")
        run = 0
        for i in range(num_runs):  
            run += 1   #count which run we are on 
            tp.start("one run, create_population_of_players")
            #initialise population of player objects, each with their own RNs
            population = create_population_of_players(titles, my_RNG, population_config)#, num_runs, destination_folder)
            tp.end("one run, create_population_of_players")


            tp.start("one run, run_one_population_episode_with_selection")
            if record_Qvalues == 'both':
                if record_Loss == 'both':
                    global_history, history_loss_dilemma, history_Loss_selection, history_Qvalues_dilemma_local, history_Qvalues_selection_local = run_one_population_episode_with_selection(run, population, num_iterations, my_RNG, destination_folder, method, population_config)
                elif record_Loss in ['dilemma', 'selection']: #if recoding only one of the losses
                    global_history, history_loss, history_Qvalues_dilemma_local, history_Qvalues_selection_local = run_one_population_episode_with_selection(run, population, num_iterations, my_RNG, destination_folder, method, population_config)
                else: #if not recording Loss 
                    global_history, empty, history_Qvalues_dilemma_local, history_Qvalues_selection_local = run_one_population_episode_with_selection(run, population, num_iterations, my_RNG, destination_folder, method, population_config)
            elif record_Loss == 'both': #not recording Qvalues but recording Loss
                global_history, history_loss_dilemma, history_Loss_selection, empty, empty = run_one_population_episode_with_selection(run, population, num_iterations, my_RNG, destination_folder, method, population_config)
            elif record_Loss in ['dilemma', 'selection']: #if recoding only one of the losses
                global_history, history_loss, empty, empty = run_one_population_episode_with_selection(run, population, num_iterations, my_RNG, destination_folder, method, population_config)
            else: #if not recording Qvalues either 
                global_history, empty, empty = run_one_population_episode_with_selection(run, population, num_iterations, my_RNG, destination_folder, method, population_config)
            print(f'finished run {run}, population {titles}')
            tp.end("one run, run_one_population_episode_with_selection")

            tp.start("one run, save_history")
            save_history(history_df=global_history, run_idx=run, destination_folder=destination_folder)
            print(f'saved history for run {run}, population {titles}')
            tp.start("one run, save_history")

            tp.start("one run, os.makedirs /LOSS & QVALUES")
            print('saving Q-values and Loss')
            if record_Loss: 
                if not os.path.isdir(str(destination_folder) + '/LOSS'):
                    os.makedirs(str(destination_folder) + '/LOSS')

            if record_Qvalues:
                if not os.path.isdir(str(destination_folder) + '/QVALUES'):
                    os.makedirs(str(destination_folder) + '/QVALUES')
            tp.end("one run, os.makedirs /LOSS & QVALUES")


            tp.start("one run, store LOSS & QVALUES")
            for idx in range(population_size): #for every player in the population 
                    #save Q_VALUES_list for player1 (learning over time) to .npy file
                tp.start("    one run, store QVALUES")
                if record_Qvalues == 'both':
                    #reduce number of Qvalues to store if parameter set to do so - TO DO fix this for dict format - how to do when every 10th Qvalue may be for a different state? 
                    #if record_Qvalues_every > 1:
                    #    history_Qvalues_dilemma_local[f'player{idx}'] = history_Qvalues_dilemma_local[f'player{idx}'][::record_Qvalues_every]
                    #    history_Qvalues_selection_local[f'player{idx}'] = history_Qvalues_selection_local[f'player{idx}'][::record_Qvalues_every]


                    #save dilemma Q_VALUES
                    if f'player{idx}' in history_Qvalues_dilemma_local.keys():
                        with open(str(destination_folder) + f'/QVALUES/Q_VALUES_dilemma_local_player{idx}_list_run{run}.pickle', 'wb') as handle:
                            pickle.dump(history_Qvalues_dilemma_local[f'player{idx}'], handle, protocol=pickle.HIGHEST_PROTOCOL)

                        ## save Q_VALUES_list for each player (learning over time) to txt file
                        #with open(str(destination_folder) + f'/QVALUES/Q_VALUES_dilemma_local_player{idx}_list_run{run}.txt', 'w') as fp:
                        #    for item in history_Qvalues_dilemma_local[f'player{idx}']:
                        #        fp.write("%s\n" % str(item))
                    #save selection Q_VALUES
                    if f'player{idx}' in history_Qvalues_selection_local.keys():
                        with open(str(destination_folder) + f'/QVALUES/Q_VALUES_selection_local_player{idx}_list_run{run}.pickle', 'wb') as handle:
                            pickle.dump(history_Qvalues_selection_local[f'player{idx}'], handle, protocol=pickle.HIGHEST_PROTOCOL)

                        ## save Q_VALUES_list for each player (learning over time) to txt file
                        #with open(str(destination_folder) + f'/QVALUES/Q_VALUES_selection_local_player{idx}_list_run{run}.txt', 'w') as fp:
                        #    for item in history_Qvalues_selection_local[f'player{idx}']:
                        #        fp.write("%s\n" % str(item))
                tp.end("    one run, store QVALUES")


                    #save loss for player1 (learning over time) to .npy file
                tp.start(f"    one run, store {record_Loss} LOSS")
                if record_Loss == 'both': 
                    #store separately history_loss_dilemma, history_Loss_selection
                    #save dilemma Loss 
                    if f'player{idx}' in history_loss_dilemma.keys():
                        with open(str(destination_folder) + f'/LOSS/LOSS_dilemma_player{idx}_list_run{run}.pickle', 'wb') as handle:
                            pickle.dump(history_loss_dilemma[f'player{idx}'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                        #save loss for each player (learning over time) to txt file
                        with open(str(destination_folder) + f'/LOSS/LOSS_dilemma_player{idx}_list_run{run}.txt', 'w') as fp:
                            for item in history_loss_dilemma[f'player{idx}']:
                                fp.write("%s\n" % str(item))
                    #save selection Loss 
                    if f'player{idx}' in history_Loss_selection.keys():
                        with open(str(destination_folder) + f'/LOSS/LOSS_selection_player{idx}_list_run{run}.pickle', 'wb') as handle:
                            pickle.dump(history_Loss_selection[f'player{idx}'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                        ## save loss for each player (learning over time) to txt file
                        #with open(str(destination_folder) + f'/LOSS/LOSS_selection_player{idx}_list_run{run}.txt', 'w') as fp:
                        #    for item in history_Loss_selection[f'player{idx}']:
                        #        fp.write("%s\n" % str(item))

                elif record_Loss in ['dilemma', 'selection']: 
                    if f'player{idx}' in history_loss.keys():
                        with open(str(destination_folder) + f'/LOSS/LOSS_player{idx}_list_run{run}.pickle', 'wb') as handle:
                            pickle.dump(history_loss[f'player{idx}'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                        ## save loss for each player (learning over time) to txt file
                        #with open(str(destination_folder) + f'/LOSS/LOSS_player{idx}_list_run{run}.txt', 'w') as fp:
                        #    for item in history_loss[f'player{idx}']:
                        #        fp.write("%s\n" % str(item))
                            #print('done saving loss for player1')                    
                tp.end(f"    one run, store {record_Loss} LOSS")

                print(f'saved {record_Loss} loss & {record_Qvalues} Q-values for run {run}, player {idx}, population {titles}')
                tp.end("one run, store LOSS & QVALUES")

        tp.end("all runs in num_runs")
#TO DO  save Loss for 'both'  


#also define general functions - used across study types 

def store_population_learning_parameters(destination_folder, method, titles, population_config): 
    """ Use this in the population case.
     This function stores the learning parameters to disc. """
    
    #alpha0, decay, gamma are defined outside of this function!!  #NOTE TO DO check that externally defined variables are ok to be used inside functions   

    population_size = population_config['population_size']
    selection_size = population_config['selection_size']
    selection_state_size = population_config['selection_state_size']
    state_size = population_config['state_size']

    #on the first run, save QL parameters to file for future reference
    with open(f'{destination_folder}/population.txt', 'a') as fp:
        fp.write("Population of players, {index : player type}: \n")
        fp.write("%s \n" % dict(zip(range(len(titles)), titles)))
        print('saved population')
        
    with open(f'{destination_folder}/QL_parameters.txt', 'w') as fp:
        fp.write("Population size: %s \n" % str(population_size))
        fp.write("Hence selecting partner out of: %s \n" % str(selection_size))
        fp.write(f"With dilemma state_size {state_size} and action_size {action_size}, and history length 1 (observations of one previous pair of moves when selecting a partner), this results in {(action_size*state_size_game)**(population_size-1)} possible selection states \n"
                 )

        fp.write("The Q-learning parameters are:     \n") # write each item on a new line
        
        fp.write("discount factor gamma: %s \n" % str(gamma))
        fp.write("learning method: %s \n" % str(method))

        fp.write("1. Dilemma Game parameters: \n")
        fp.write("    initial exploration rate eps0: %s \n" % str(eps_dil_initial)) 
        fp.write("    Is eps (exploration) decay to 0 present: %s \n" % str(epsdecay))
        fp.write("    final exploration rate eps: %s \n" % str(eps_dil_final)) 


        fp.write("2. Partner Selection parameters: \n")
        fp.write("    initial exploration rate eps0: %s \n" % str(eps_sel_initial)) 
        fp.write("    Is eps (exploration) decay to 0 present: %s \n" % str(epsdecay))
        fp.write("    final exploration rate eps: %s \n" % str(eps_sel_final)) 

        if method=='DQN':
            fp.write("\n")
            fp.write("player2_learns: %s \n"%str(player2_learns))
            fp.write("dilemma_state_uses_identity: %s \n"%str(dilemma_state_uses_identity))

            fp.write("\n DQN parameters: \n")
            fp.write("    LR for gradient descent, SELECTION net =%s \n" % str(LR_selection))
            fp.write("    LR for gradient descent, DILEMMa net =%s \n" % str(LR_dilemma))
            fp.write("    TAU=%s \n" % str(TAU))
            fp.write("    UPDATE_EVERY_selection=%s, \n" % str(UPDATE_EVERY_selection))
            fp.write("    UPDATE_EVERY_dilemma=%s, \n" % str(UPDATE_EVERY_dilemma))
            fp.write("    BUFFER_SIZE=%s, \n" %str(BUFFER_SIZE))
            #fp.write("    BATCH_SIZE=%s \n" %str(BATCH_SIZE))
            fp.write("    network_size=%s \n" %str(network_size))
            fp.write("    use_target_network=%s \n" %str(use_target_network))
        else: 
            fp.write("initial learning rate alpha0 for tabular Q-learning: %s \n" % str(alpha0)) 
            fp.write("learning rate decay for tabular Q-learning: %s \n" % str(decay))

        fp.write("\n \n")
        fp.write("record_Qvalues: %s \n" % str(record_Qvalues))
        #fp.write("record_Qvalues_every: %s iterations \n" % str(record_Qvalues_every))
        fp.write("record_Loss: %s \n" % str(record_Loss))

        print('saved parameters for population')

            #save population to disc


    #return initial_selection_states


def store_network_architecture(counter, destination_folder, summary):
    if counter == 1: 
        with open(f'{destination_folder}/QL_parameters.txt', 'a') as fp:
            fp.write("DQN architecture: %s \n" % str(summary))
            #fp.write("DQN architecture: \n")
            #for param_tensor in summary:
            #    fp.write("%s \n" % str(param_tensor, "\t", int(summary[param_tensor].size())))


def create_population_of_players(titles, my_RNG, population_config):#, num_runs, destination_folder):
    '''use in the population case. 
    This function creates a population of players for a single run.'''

    population_size = population_config['population_size']
    selection_size = population_config['selection_size']
    selection_state_size = population_config['selection_state_size']
    state_size = population_config['state_size']
    
    population = []
    idx = -1
    for x in range(len(titles)): #loop over key, value pairs in dict #TO DO understand if this is done in order
        idx += 1 
        title = titles[x]
        #initialise each player in the population using DQN_weights seed and memory_sampling seed from the RN streams
        int_x1 = int(my_RNG.player_streams[f'player{idx}'][5].integers(low=1,high=100, size=1)[0])
        int_x2 = int(my_RNG.player_streams[f'player{idx}'][6].integers(low=1,high=100, size=1)[0])
        int_x3 = int(my_RNG.player_streams[f'player{idx}'][10].integers(low=1,high=100, size=1)[0])
        int_x4 = int(my_RNG.player_streams[f'player{idx}'][11].integers(low=1,high=100, size=1)[0])

        population.append(Player(strategy=strategy_mapping[title], moral_type=moral_mapping[title], 
                                 population_size=population_size, selection_size=selection_size, selection_state_size=selection_state_size, state_size=state_size,
                                 dil_DQN_weights_seed = int_x1, dil_memory_sampling_seed = int_x2, sel_DQN_weights_seed = int_x3, sel_memory_sampling_seed = int_x4))

    return population


def save_history(history_df, run_idx, destination_folder):
    if not os.path.isdir(str(destination_folder)+'/history'):
        os.makedirs(str(destination_folder)+'/history')
    #history_df = history_df.drop(columns=["selection_state_player1", "selection_state_player2","selection_next_state_player1","selection_next_state_player2","eps_selection_player1","eps_player1","eps_player2"])
    history_df.to_csv(str(destination_folder)+f'/history/run{run_idx}.csv')

 
# initialise time profiler
tp = TimeProfiler(profile_time = profile_time)

#tp.start("start test runs")
#testing = False #overwrite the parameter from config here 
if testing: 
    #RUN TESTS - 14 & 27 June 2023 - one run 
    population_size = 6 #21
    #create global arguments usnig population_size from above 
    selection_size = population_size-1 #number of possible partner selections a playercan make 
    if state_pairs: selection_state_size = action_size * selection_size #length of the selection state - a set of (population_size-1) pairs of moves 
    else: selection_state_size = selection_size   
    if dilemma_state_uses_identity: state_size = state_size_game + (population_size-1)
    else: state_size = state_size_game
    population_config = {'population_size':population_size, 'selection_size':selection_size, 'selection_state_size':selection_state_size, 'state_size':state_size}
    #run_population_and_save_withselection(num_runs=1, num_iterations=1000, destination_folder='3xS_TEST', titles=['QLS', 'QLS', 'QLS'], method='DQN', population_config=population_config)
    #run_population_and_save_withselection(num_runs=1, num_iterations=1000, destination_folder='1xS_1xUT_1xDE_1xVEe_1xVEk_1xAL_TEST', titles=['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k', 'QLAL'], method='DQN', population_config=population_config)
    #run_population_and_save_withselection(num_runs=1, num_iterations=100000, destination_folder='1xS_1xAC_1xAD', titles=['QLS', 'AC', 'AD'], method='DQN', population_config=population_config)
    #run_population_and_save_withselection(num_runs=1, num_iterations=10000, destination_folder='3xS_TEST', titles=['QLS', 'QLS', 'QLS'], method='DQN', population_config=population_config)
    #run_population_and_save_withselection(num_runs=1, num_iterations=1000, destination_folder='2xS_TEST', titles=['QLS', 'QLS'], method='DQN', population_config=population_config)
    run_population_and_save_withselection(num_runs=1, num_iterations=5, destination_folder='1xaUT_1xmDE_1xVEie_1xVEagg_1xaAL_TEST', titles=['QLS', 'QLaUT', 'QLmDE', 'QLVie', 'QLVagg', 'QLaAL'], method='DQN', population_config=population_config)

    #run_population_and_save_withselection(num_runs=1, num_iterations=1000, destination_folder='1xS_1xAC_1xAD_TEST', titles=['QLS', 'AC', 'AD'], method='DQN', population_config=population_config)
    #run_population_and_save_withselection(num_runs=1, num_iterations=1000, destination_folder='1xS_10xAC_10xAD_TEST', titles=['QLS', 'AC', 'AC', 'AC', 'AC', 'AC', 'AC', 'AC', 'AC', 'AC', 'AC', 'AD', 'AD', 'AD', 'AD', 'AD', 'AD', 'AD', 'AD', 'AD', 'AD'], method='DQN', population_config=population_config)
    #run_population_and_save_withselection(num_runs=1, num_iterations=100, destination_folder='1xS_1xCD_TEST', titles=['QLS', 'CD'], method='DQN', population_config=population_config)
    #run_population_and_save_withselection(num_runs=1, num_iterations=100, destination_folder='1xS_1xGEa_1xGEb_1xGEc', titles=['QLS', 'GEa', 'GEb', 'GEc'], method='DQN', population_config=population_config)
#tp.end("start test runs")



tp.start("define parsing for all arguments")
parser = argparse.ArgumentParser(description='Process two player titles (short versions) from user string input.')
parser.add_argument('--destination_folder', type=str, required=True, help='the set of players & destinaton_folder name - required')
#parser.add_argument('--population_size', type=int, required=True, help='the number of players in the population - optional, default 5')
parser.add_argument('--method', type=str, required=False, help='method of learning - tabularQ or DQN - optional, default DQN')
#parser.add_argument('--master_seed', type=int, required=False, help='master seed to initialise random number generator - optional, default 1')
parser.add_argument('--num_iterations', type=int, required=False, help='number of iterations to be run in one run; in the episodes study, this identifies number of episodes NOT individual iterations - optional, default 30000')
parser.add_argument('--num_runs', type=int, required=False, help='number of runs/replicas with different seeds to be created - optional, default 20')
#parser.add_argument('--eps0', type=float, required=False, help='the initial exploration rate eps to be used when playing actions (will decay to 0) - optional, default 0.05. NOTE if you feed in eps0, it will assume that eps must decay to 0')
#parser.add_argument('--epsdecay', type=bool, required=False, help='whether eps decay is present, default False')
#parser.add_argument('--gamma', type=float, required=False, help='the discount factor to be used in Q-Learning, default 0.9')
#parser.add_argument('--beta', type=float, required=False, help='the beta to be used for relative weighting of two mixed virtue rewards, default 0.5')
parser.add_argument('--extra', type=str, required=False, help='extra detail to be added to the destination_folder name - optional')
#parser.add_argument('--LR', type=str, required=False, help='learning rate LR to be used in DQN stochastic gradient descent, default 0.01')
#parser.add_argument('--alpha0', type=float, required=False, help='the initial learning rate alpha to be used in Q-Learning, default 0.01')
#parser.add_argument('--decay', type=float, required=False, help='the learning rate decay to be used in Q-Learning, default 0')
tp.end("define parsing for all arguments")

tp.start("parser.parse_args()")
args = parser.parse_args()
tp.end("parser.parse_args()")

destination_folder = args.destination_folder

tp.start("create titles")
titles = [] 
population_size = 0
for item in destination_folder.split('_'): 
    items = item.split('x')
    population_size += int(items[0])
    for agent in range(int(items[0])):
        if items[1] not in ['AC','AD','TFT','Random','CD','GEa','GEb','GEc', 'GEd', 'GEe']: #if not one of the pre-defined fixed-strategy players 
            titles.append(str('QL'+items[1]).replace('VEe', 'VE_e').replace('VEk', 'VE_k'))
        else: 
            titles.append(str(items[1]))
tp.end("create titles")
print('running population_size: '+ str(population_size))


#create global arguments usnig population_size from above 
selection_size = population_size-1 #number of possible partner selections a playercan make 
if state_pairs: 
    selection_state_size = action_size * selection_size #length of the selection state - a set of (population_size-1) pairs of moves 
else: 
    selection_state_size = selection_size
if dilemma_state_uses_identity:
    state_size = state_size_game + (population_size-1)
else:
    state_size = state_size_game

population_config = {'population_size':population_size, 'selection_size':selection_size, 'selection_state_size':selection_state_size, 'state_size':state_size}


destination_folder = destination_folder + '__'

#if args.population_size:
#    population_size = args.population_size
#    #destination_folder += f'_popsize{population_size}'
#else: 
#    population_size = 5

tp.start("read in optional arguments")
#try to read in the optional arguments
if args.method:
    method = args.method
    destination_folder += f'_method{method}'
else: 
    method = 'DQN'


if args.num_iterations:
    num_iterations = args.num_iterations
    destination_folder += f'_iter{num_iterations}'
else: 
    num_iterations=30000 

if args.num_runs:
    num_runs = args.num_runs
    destination_folder += f'_runs{num_runs}'
else: 
    num_runs = 20

if args.extra:
    destination_folder += f'_{args.extra}'

if False: #if not using a config file 
    if args.master_seed:
        master_seed = args.master_seed
        destination_folder += f'_seed{master_seed}'
    else: 
        master_seed = 1 

    if args.eps0:
        eps0 = args.eps0
        destination_folder += f'_eps0{eps0}'
    else: 
        eps0 = 0.05

    if args.epsdecay:
        epsdecay = True
        destination_folder += '_epsdecay'
    else: 
        epsdecay = False

    if args.gamma:
        gamma = args.gamma
        destination_folder += f'_gamma{gamma}'
    else: 
        gamma = 0.9 

    if args.beta:
        mixed_beta = args.beta
        destination_folder += f'_beta{mixed_beta}'
    else: 
        mixed_beta = 0.5

    if args.LR:
        LR = args.LR
        destination_folder += f'_LR{LR}'
    else: 
        LR = 0.01

if False: #for tabular q-learning
    if args.alpha0:
        alpha0 = args.alpha0
        destination_folder += f'_alpha0{alpha0}'
    else: 
        alpha0 = 0.01

    if args.decay:
        decay = args.decay
        destination_folder += f'_decay{decay}'
    else: 
        decay = 0 

tp.end("read in optional arguments")


tp.start("run_population_and_save_withselection")
run_population_and_save_withselection(num_runs=num_runs, num_iterations=num_iterations, destination_folder=destination_folder, titles=titles, method=method, population_config=population_config)
tp.end("run_population_and_save_withselection")

tp.summarise('results/'+destination_folder + "/time_profile_summary.txt")


#### with Random Matching (no Partner Selection) ####
if False: 
    run_population_and_save(num_runs=num_runs, num_iterations=num_iterations, destination_folder=destination_folder, titles=titles, method=method)
#TO DO after establishing which function to use for this pair of players, run given number of episodes & iterations for this specific pair: 


#### older #### 
if False: 
    #read in two player titles from user input on the command line
    parser = argparse.ArgumentParser(description='Process two player titles (short versions) from user string input.')
    parser.add_argument('--title1', type=str, required=True, help='the title for player1 - required')
    parser.add_argument('--title2', type=str, required=True, help='the title for player2 - required')

    parser.add_argument('--master_seed', type=int, required=False, help='master seed to initialise random number generator - optional, default 1')
    parser.add_argument('--num_iterations', type=int, required=False, help='number of iterations to be run in one run - optional, default 10000')
    parser.add_argument('--num_runs', type=int, required=False, help='number of runs/replicas with different seeds to be created - optional, default 100')
    parser.add_argument('--eps0', type=float, required=False, help='the initial exploration rate eps to be used when playing actions (will decay to 0) - optional, default 0.05. NOTE if you feed in eps0, it will assume that eps must decay to 0')
    parser.add_argument('--epsdecay', type=bool, required=False, help='whether eps decay is present, default False')
    parser.add_argument('--alpha0', type=float, required=False, help='the initial learning rate alpha to be used in Q-Learning, default 0.01')
    parser.add_argument('--decay', type=float, required=False, help='the learning rate decay to be used in Q-Learning, default 0')
    parser.add_argument('--gamma', type=float, required=False, help='the discount factor to be used in Q-Learning, default 0.9')
    parser.add_argument('--beta', type=float, required=False, help='the beta to be used for relative weighting of two mixed virtue rewards, default 0.5')
    parser.add_argument('--extra', type=str, required=False, help='extra detail to be added to the destination_folder name - optional')
    parser.add_argument('--LR', type=str, required=False, help='learning rate LR to be used in DQN stochastic gradient descent, default 0.01')

    args = parser.parse_args()
    title1 = args.title1
    title2 = args.title2
    destination_folder = title1+'_'+title2

    #try to read in the optional arguments
    if args.master_seed:
        master_seed = args.master_seed
        destination_folder += f'_seed{master_seed}'
    else: 
        master_seed = 1 

    if args.num_iterations:
        num_iterations = args.num_iterations
        destination_folder += f'_iter{num_iterations}'
    else: 
        num_iterations=10000 

    if args.num_runs:
        num_runs = args.num_runs
        destination_folder += f'_runs{num_runs}'
    else: 
        num_runs = 100

    if args.extra:
        destination_folder += f'_{args.extra}'


    if args.eps0:
        eps0 = args.eps0
        destination_folder += f'_eps0{eps0}'
    else: 
        eps0 = 0.05

    if args.epsdecay:
        epsdecay = True
        destination_folder += '_epsdecay'
    else: 
        epsdecay = False

    if args.alpha0:
        alpha0 = args.alpha0
        destination_folder += f'_alpha0{alpha0}'
    else: 
        alpha0 = 0.01

    if args.decay:
        decay = args.decay
        destination_folder += f'_decay{decay}'
    else: 
        decay = 0 

    if args.gamma:
        gamma = args.gamma
        destination_folder += f'_gamma{gamma}'
    else: 
        gamma = 0.9 

    if args.beta:
        mixed_beta = args.beta
        destination_folder += f'_beta{mixed_beta}'
    else: 
        mixed_beta = 0.5

    if args.LR:
        LR = args.LR
        destination_folder += f'_LR{LR}'
    else: 
        LR = 0.01

        

    #after establishing which function to use for this pair of players, run given number of episodes & iterations for this specific pair: 
    if 'QL' in title1: 
        if 'QL' in title2: 
            run_and_save(master_seed, num_runs, num_iterations, destination_folder, title1, title2)
        else: #if player2 is static
            run_mixed_and_save(master_seed, num_runs, num_iterations, destination_folder, title1, title2)
    else: #if both players are static
        run_static(master_seed, num_runs, num_iterations, destination_folder, title1, title2)

