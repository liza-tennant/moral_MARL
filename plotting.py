#plotting rewards (mean and cumulative mean with SD); actions (C or D) and actions types (in response to state) 
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.pyplot import figure
import dataframe_image as dfi
import os 
import glob 
from collections import Counter 
import pickle 
#import pickle5 as pickle #for earlier versions of python
from ast import literal_eval as make_tuple
from ordered_set import OrderedSet

sns.set(font_scale=2)

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

from statistics import mean 
import networkx as nx
from math import isnan
from collections import Counter

    

#import scipy.stats as st

#NOTE some of these functions save the plot to disc, others do not 

#TO DO 
# edit plot_last_20_actions & plot_firt_20_actions to have more readable titles and conssitent colors for certsin action-state pairs

# - add CI instead of SD to the collective outcome plots - CHECK
# - ensure QLS_TFT and QLS_Random are being plotted on collective outcomes plots 
# make colors more consistent 
# make cumulative plots y-axis all the same scale, for ease of comparison 


#PAYOFFMAT_IPD = [ [(3,3),(1,4)] , [(4,1),(2,2)] ] #IPD game 
#PAYOFFMAT_VOLUNTEER = [ [(4,4),(2,5)] , [(5,2),(1,1)] ] #VOLUNTEER game 
#PAYOFFMAT_STAGHUNT = [ [(5,5),(1,4)] , [(4,1),(2,2)] ] #STAGHUNT game 

#os.getcwd()
#os.chdir('~/Library/Mobile\ Documents/com~apple~CloudDocs/PhD_data') 

####################################
#### globally useful variables ####
####################################


title_mapping = {'AC':'AlwaysCooperate', 'AD':'AlwaysDefect', 'TFT':'TitForTat', 'Random':'random', 
                 'CD':'Alternating', 'GEa':'GilbertElliot_typea', 'GEb':'GilbertElliot_typeb', 'GEc':'GilbertElliot_typec', 'GEd':'GilbertElliot_typed', 'GEe':'GilbertElliot_typee',
                 'QLS':'Selfish', 'QLUT':'Utilitarian', 'QLDE':'Deontological', 'QLVE_e':'VirtueEthics_equality', 'QLVE_k':'VirtueEthics_kindness', 'QLVM':'VirtueEthics_mixed', 'QLAL':'Altruist',
                 'QLaUT':'anti-Utilitarian', 'QLmDE':'malicious_Deontological', 'QLVie':'VirtueEthics_inequality', 'QLVagg':'VirtueEthics_aggressiveness', 'QLaAL':'anti-Altruist'}

if False: #previous versions 
    color_mapping = {'QLS':'orange', 'AC':'green', 'AD':'orchid', 'TFT':'orangered', 'Random':'grey', 
                    'CD':'deeppink', 'GEa':'limegreen', 'GEb':'forestgreen', 'GEc':'darkgreen', 'GEd':'magenta', 'GEe':'lightgreen',
                    'QLUT':'blue', 'QLDE':'turquoise', 'QLVE_e':'mediumorchid', 'QLVE_k':'forestgreen'}

    #color_mapping_longtitle = {'Selfish':'orange', 'AlwaysCooperate':'green', 'AlwaysDefect':'orchid', 'TitfroTat':'orangered', 'Random':'grey', 
    #                 'Alternating':'deeppink', 'GilbertElliot_typea':'limegreen', 'GilbertElliot_typeb':'forestgreen', 'GilbertElliot_typec':'springgreen', 'GilbertElliot_typed':'magenta', 'GilbertElliot_typee':'lightgreen', 
    #                 'Utilitarian':'blue', 'Deontological':'turquoise', 'VirtueEthics_equality':'mediumorchid', 'VirtueEthics_kindness':'forestgreen',
    #                 'anti-Utilitarian':'cyan', 'malicious_Deontological':'teal', 'VirtueEthics_inequality':'purple', 'VirtueEthics_aggressiveness':'limegreen'}
                    
color_mapping_longtitle = {'Selfish':'orange', 'AlwaysCooperate':'green', 'AlwaysDefect':'orchid', 'TitfroTat':'orangered', 'Random':'grey', 
                           'Alternating':'deeppink', 'GilbertElliot_typea':'limegreen', 'GilbertElliot_typeb':'forestgreen', 'GilbertElliot_typec':'springgreen', 'GilbertElliot_typed':'magenta', 'GilbertElliot_typee':'lightgreen', 
			               'Utilitarian':'cornflowerblue', 'Deontological':'aqua', 'VirtueEthics_equality':'mediumorchid', 'VirtueEthics_kindness':'lime',
                            'anti-Utilitarian':'blue', 'malicious_Deontological':'lightseagreen', 'VirtueEthics_inequality':'purple', 'VirtueEthics_aggressiveness':'forestgreen'}
                 
movingaverage = True 
movingN = 200 #overwrite this as needed 
n_runs_global = 20 



population_size_approx = 16 

title_mapping_forpaper_long = {'AlwaysCooperate':'AC', 'AlwaysDefect':'AD', 'TitForTat':'TFT', 'random':'Random', 
                 'Alternating':'CD', 'GilbertElliot_typea':'GEa', 'GilbertElliot_typeb':'GEb', 'GilbertElliot_typec':'GEc', 'GilbertElliot_typed':'GEd', 'GilbertElliot_typee':'GEe',
                 'Selfish':'S', 'Utilitarian':'Ut', 'Deontological':'De', 'VirtueEthics_equality':'V-Eq', 'VirtueEthics_kindness':'V-Ki',
                 'anti-Utilitarian':'aUt', 'malicious_Deontological':'mDe', 'VirtueEthics_inequality':'V-In', 'VirtueEthics_aggressiveness':'V-Ag'}


title_mapping_forpaper_short = {'AC':'AC', 'AD':'AD', 'TFT':'TFT', 'Random':'Random', 
                 'CD':'Al', 'GEa':'GEa', 'GEb':'GEb', 'GEc':'GEc', 'GEd':'GEd', 'GEe':'GEe',
                 'QLS':'S', 'QLUT':'Ut', 'QLDE':'De', 'QLVE_e':'V-Eq', 'QLVE_k':'V-Ki',
                 'QLaUT':'aUt', 'QLmDE':'mDe', 'QLVie':'V-In', 'QLVagg':'V-Ag'}

title_mapping_forpaper_shorttolong  = {v: k for k, v in title_mapping_forpaper_long.items()}


title_mapping_forpaper_longtoclean = {'AlwaysCooperate':'AC', 'AlwaysDefect':'AD', 'TitForTat':'TFT', 'random':'Random', 
                 'Alternating':'CD', 'GilbertElliot_typea':'GEa', 'GilbertElliot_typeb':'GEb', 'GilbertElliot_typec':'GEc', 'GilbertElliot_typed':'GEd', 'GilbertElliot_typee':'GEe',
                 'Selfish':'Selfish', 'Utilitarian':'Utilitarian', 'Deontological':'Deontological', 'VirtueEthics_equality':'Virtue-Equality', 'VirtueEthics_kindness':'Virtue-Kindness',
                 'anti-Utilitarian':'anti-Utilitarian', 'malicious_Deontological':'malicious-Deontological', 'VirtueEthics_inequality':'Virtue-Inequality', 'VirtueEthics_aggressiveness':'Virtue-Aggressive'}


sns.set(font_scale=2)


########################
#### pairwise plots ####
########################
def plot_results(destination_folder, player1_title, player2_title, n_runs, game_title):
    '''plot reward over time - cumulative and non-cumulative; game reward and intrinsic and others'''
    if not os.path.isdir(str(destination_folder)+'/plots'):
        os.makedirs(str(destination_folder)+'/plots')

    if not os.path.isdir(str(destination_folder)+'/plots/outcome_plots'):
        os.makedirs(str(destination_folder)+'/plots/outcome_plots')

    ##################################
    #### cumulative - game reward ####
    ##################################
    my_df_player1 = pd.read_csv(f'{destination_folder}/player1/df_cumulative_reward_game.csv', index_col=0)
    means_player1 = my_df_player1.mean(axis=1)
    sds_player1 = my_df_player1.std(axis=1)
    #ci = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    ci_player1 = 1.96 * sds_player1/np.sqrt(n_runs)

    my_df_player2 = pd.read_csv(f'{destination_folder}/player2/df_cumulative_reward_game.csv', index_col=0)
    means_player2 = my_df_player2.mean(axis=1)
    sds_player2 = my_df_player2.std(axis=1)
    ci_player2 = 1.96 * sds_player2/np.sqrt(n_runs)

    #plt.figure(dpi=80) #figsize=(10, 6) 
    plt.figure(dpi=80, figsize=(5, 4))
    plt.rcParams.update({'font.size':20})
    plt.plot(my_df_player1.index[:], means_player1[:], label=f'player1 - {player1_title}', color='blue')
    plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
    #plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
    #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
    plt.fill_between(my_df_player1.index[:], means_player1-ci_player1, means_player1+ci_player1, facecolor='#95d0fc', alpha=0.7)
    plt.fill_between(my_df_player2.index[:], means_player2-ci_player2, means_player2+ci_player2, facecolor='#fed8b1', alpha=0.7)
    
    plt.title('Cumulative Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title+' vs '+player2_title)
    #plt.title('Cumulative Game Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
    plt.ylabel('Cumulative Game reward')
    plt.xlabel('Iteration')
    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    plt.savefig(f'{destination_folder}/plots/reward_cumulative_Game.png', bbox_inches='tight')


    ######################################
    #### non-cumulative - game reward ####
    ######################################
    my_df_player1 = pd.read_csv(f'{destination_folder}/player1/df_reward_game.csv', index_col=0)
    means_player1 = my_df_player1.mean(axis=1)
    sds_player1 = my_df_player1.std(axis=1)
    ci_player1 = 1.96 * sds_player1/np.sqrt(n_runs)

    my_df_player2 = pd.read_csv(f'{destination_folder}/player2/df_reward_game.csv', index_col=0)
    means_player2 = my_df_player2.mean(axis=1)
    sds_player2 = my_df_player2.std(axis=1)
    ci_player2 = 1.96 * sds_player2/np.sqrt(n_runs)

    plt.figure(dpi=80) #figsize=(10, 6) 
    plt.plot(my_df_player1.index[:], means_player1[:], lw=0.5, label=f'player1 - {player1_title}', color='blue')
    plt.plot(my_df_player2.index[:], means_player2[:], lw=0.5, label=f'player2 - {player2_title}', color='orange')
    #plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
    #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
    plt.fill_between(my_df_player1.index[:], means_player1-ci_player1, means_player1+ci_player1, facecolor='#95d0fc', alpha=0.7)
    plt.fill_between(my_df_player2.index[:], means_player2-ci_player2, means_player2+ci_player2, facecolor='#fed8b1', alpha=0.7)
    plt.title('Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title+' vs '+player2_title)
    #plt.title(r'Game Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
    #plt.gca().set_ylim([1, 4]) #[0,5]
    if game_title=='IPD':
        plt.gca().set_ylim([1, 4])
    elif game_title=='VOLUNTEER':
        plt.gca().set_ylim([1, 5])
    elif game_title=='STAGHUNT':  
       plt.gca().set_ylim([1, 5])  
    plt.ylabel('Game reward')
    plt.xlabel('Iteration')
    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    plt.savefig(f'{destination_folder}/plots/reward_Game.png', bbox_inches='tight')

    if False: #do not plot at the moment
        plt.figure(figsize=(10, 6), dpi=80)
        plt.plot(my_df_player1.index[:], means_player1[:], lw=0.5, label=f'player1 - {player1_title}', color='blue')
        plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
        plt.title('Game Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+player1_title)
        plt.gca().set_ylim([0, 5])
        plt.ylabel('Game reward')
        plt.xlabel('Iteration')
        plt.legend(loc='upper left')
        plt.savefig(f'{destination_folder}/plots/reward_Game_player1.png')


        plt.figure(figsize=(10, 6), dpi=80)
        plt.plot(my_df_player2.index[:], means_player2[:], lw=0.5, label=f'player2 - {player2_title}', color='orange')
        plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
        plt.title(r'Game Reward (Mean over '+str(n_runs)+r' runs $\pm$ SD), '+player2_title)
        plt.gca().set_ylim([0, 5])
        plt.ylabel('Game reward')
        plt.xlabel('Iteration')
        plt.legend(loc='upper left')
        plt.savefig(f'{destination_folder}/plots/reward_Game_player2.png')




    ##################################
    #### cumulative - intrinsic reward ###
    ##################################
    my_df_player1 = pd.read_csv(f'{destination_folder}/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
    means_player1 = my_df_player1.mean(axis=1)
    sds_player1 = my_df_player1.std(axis=1)

    my_df_player2 = pd.read_csv(f'{destination_folder}/player2/df_cumulative_reward_intrinsic.csv', index_col=0)
    means_player2 = my_df_player2.mean(axis=1)
    sds_player2 = my_df_player2.std(axis=1)

    #plot player1 and player2 separately
    #only plot player1 intrinsic reward if they are a QL player 
    if 'QL' in player1_title:
        if 'Selfish' not in player1_title: 
            plt.figure(dpi=80) #figsize=(10, 6)
            plt.plot(my_df_player1.index[:], means_player1[:], label=f'player1 - {player1_title}', color='blue')
            #plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
            plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
            #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            #plt.title(r'Cumulative Intrinsic Reward (Mean over '+str(n_runs)+r' runs $\pm$ SD), '+player1_title+' vs '+player2_title)
            plt.title('Cumulative Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title)
            #plt.title(r'Cumulative Intrinsic Reward (Mean over '+str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title)
            plt.ylabel('Cumulative Intrinsic reward')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/reward_cumulative_Intrinsic_player1.png', bbox_inches='tight')

    #only plot player2 intrinsic reward if they are a QL player 
    if 'QL' in player2_title:
        if 'Selfish' not in player2_title: 
            plt.figure(dpi=80) #figsize=(10, 6)
            plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
            plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            plt.title('Cumulative Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player2_title)
            #plt.title(r'Cumulative Intrinsic Reward (Mean over '+str(n_runs)+r' runs $\pm$ SD), '+'\n'+player2_title)
            plt.ylabel('Cumulative Intrinsic reward')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/reward_cumulative_Intrinsic_player2.png', bbox_inches='tight')


    ######################################
    #### non-cumulative - intrinsic reward ###
    ######################################
    #only plot player1 intrinsic reward if they are a QL player 
    if 'QL' in player1_title:
        if 'Selfish' not in player1_title: 
            my_df_player1 = pd.read_csv(f'{destination_folder}/player1/df_reward_intrinsic.csv', index_col=0)
            means_player1 = my_df_player1.mean(axis=1)
            sds_player1 = my_df_player1.std(axis=1)

            plt.figure(dpi=80) #figsize=(10, 6)
            plt.plot(my_df_player1.index[:], means_player1[:], lw=0.5, label=f'player1 - {player1_title}', color='blue')
            #plt.plot(my_df_player2.index[:], means_player2[:], lw=0.5, label=f'player2 - {player2_title}', color='orange')
            plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
            #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            plt.title('Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title)
            #plt.title(r'Intrinsic Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title)
            if game_title=='IPD':
                plt.gca().set_ylim([-5, 6])
            elif game_title=='VOLUNTEER':
                plt.gca().set_ylim([-5, 8])
            elif game_title=='STAGHUNT':  
                plt.gca().set_ylim([-5, 10])  
            plt.ylabel('Intrinsic reward')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/reward_Intrinsic_player1.png', bbox_inches='tight')

    if 'QL' in player2_title:
        if 'Selfish' not in player2_title: 
            my_df_player2 = pd.read_csv(f'{destination_folder}/player2/df_reward_intrinsic.csv', index_col=0)
            means_player2 = my_df_player2.mean(axis=1)
            sds_player2 = my_df_player2.std(axis=1)

            plt.figure(dpi=80) #figsize=(10, 6)
            plt.plot(my_df_player2.index[:], means_player2[:], lw=0.5, label=f'player2 - {player2_title}', color='orange')
            plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            plt.title('Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player2_title)
            #plt.title(r'Intrinsic Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player2_title)
            if game_title=='IPD':
                plt.gca().set_ylim([-5, 6])
            elif game_title=='VOLUNTEER':
                plt.gca().set_ylim([-5, 8])
            elif game_title=='STAGHUNT':  
                plt.gca().set_ylim([-5, 10])
            plt.ylabel('Intrinsic reward')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/reward_Intrinsic_player2.png', bbox_inches='tight')

    if False: #do not plot at the moment 
        ##################################
        #### cumulative - collective game reward ####
        ##################################
        my_df = pd.read_csv(f'{destination_folder}/df_cumulative_reward_collective.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)


        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
    #    plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        plt.title('Cumulative Collective Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title+' vs '+player2_title)
        #plt.title(r'Cumulative Collective Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        plt.ylabel('Cumulative Collective reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_cumulative_Collective.png', bbox_inches='tight')

        ######################################
        #### non-cumulative - collective game reward ####
        ######################################
        #TO DO 
        my_df = pd.read_csv(f'{destination_folder}/df_reward_collective.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
    #    plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        plt.title('Collective Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title+' vs '+player2_title)
        #plt.title(r'Collective Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        if game_title=='IPD':
            plt.gca().set_ylim([4, 6])
        elif game_title=='VOLUNTEER':
            plt.gca().set_ylim([2, 8])
        elif game_title=='STAGHUNT': 
            plt.gca().set_ylim([4, 10])
        plt.ylabel('Collective reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_Collective.png', bbox_inches='tight')


        ##################################
        #### cumulative - gini game reward ####
        ##################################
        my_df = pd.read_csv(f'{destination_folder}/df_cumulative_reward_gini.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        #plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        #plt.title('Cumulative Gini Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+'runs), '+'\n'+player1_title+' vs '+player2_title)
        plt.title(r'Cumulative Gini Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        plt.ylabel('Cumulative Gini reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_cumulative_Gini.png', bbox_inches='tight')

        ######################################
        #### non-cumulative - gini game reward ####
        ######################################
        #TO DO 
        my_df = pd.read_csv(f'{destination_folder}/df_reward_gini.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        #plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        #plt.title('Gini Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+'runs), '+'\n'+player1_title+' vs '+player2_title)
        plt.title(r'Gini Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        plt.gca().set_ylim([0, 1])
        plt.ylabel('Gini reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_Gini.png', bbox_inches='tight')




        ##################################
        #### cumulative - min game reward ####
        ##################################
        my_df = pd.read_csv(f'{destination_folder}/df_cumulative_reward_min.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        #plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        #plt.title('Cumulative Min Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+'runs), '+'\n'+player1_title+' vs '+player2_title)
        plt.title(r'Cumulative Min Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        plt.ylabel('Cumulative Min reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_cumulative_Min.png', bbox_inches='tight')

        ######################################
        #### non-cumulative - min game reward ####
        ######################################
        #TO DO 
        my_df = pd.read_csv(f'{destination_folder}/df_reward_min.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        #plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        #plt.title('Min Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+'runs), '+'\n'+player1_title+' vs '+player2_title)
        plt.title(r'Min Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        if game_title=='IPD':
            plt.gca().set_ylim([1, 3])
        elif game_title=='VOLUNTEER':
            plt.gca().set_ylim([1, 4])
        elif game_title=='STAGHUNT': 
            plt.gca().set_ylim([1, 5]) 
        plt.ylabel('Min reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_Min.png', bbox_inches='tight')


def plot_actions(destination_folder, player1_title, player2_title, n_runs):
    '''explore the actions taken by players at every step out of 10000 (or num_iter) - what percentage are cooperating? '''

    actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)
    #actions_player1.iloc[9999].value_counts()

    #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
    actions_player1['%_defect'] = actions_player1[actions_player1[:]==1].count(axis='columns')
    actions_player1['%_cooperate'] = n_runs-actions_player1['%_defect']

    #convert to %
    actions_player1['%_defect'] = (actions_player1['%_defect']/n_runs)*100
    actions_player1['%_cooperate'] = (actions_player1['%_cooperate']/n_runs)*100

    #plot results 
    plt.figure(dpi=80) #figsize=(10, 6), 
    plt.plot(actions_player1.index[:], actions_player1['%_cooperate'], label=f'player1 - {player1_title}', color='blue')
    #plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
    #plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
    #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
    plt.title('The actions of Player 1 at every step of the episode \n (percentage cooperated over '+str(n_runs)+r' runs)')
    plt.gca().set_ylim([0, 100])
    plt.ylabel('Percentage cooperating')
    plt.xlabel('Iteration')
    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    plt.savefig(f'{destination_folder}/plots/actions_player1.png', bbox_inches='tight')



    #repeat for player2
    actions_player2 = pd.read_csv(f'{destination_folder}/player2/action.csv', index_col=0)
    #actions_player2.iloc[9999].value_counts()

    actions_player2['%_defect'] = actions_player2[actions_player2[:]==1].count(axis='columns')
    actions_player2['%_cooperate'] = n_runs-actions_player2['%_defect']

    #convert to %
    actions_player2['%_defect'] = (actions_player2['%_defect']/n_runs)*100
    actions_player2['%_cooperate'] = (actions_player2['%_cooperate']/n_runs)*100

    plt.figure(dpi=80) #figsize=(10, 6), 
    plt.plot(actions_player2.index[:], actions_player2['%_cooperate'], label=f'player2 - {player2_title}', color='orange')
    #plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
    #plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
    #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
    plt.title('The actions of Player 2 at every step of the episode \n (percentage cooperated over '+str(n_runs)+r' runs)')
    plt.gca().set_ylim([0, 100])
    plt.ylabel('Percentage cooperating')
    plt.xlabel('Iteration')
    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    plt.savefig(f'{destination_folder}/plots/actions_player2.png', bbox_inches='tight')


def plot_action_types_area(destination_folder, player1_title, player2_title, n_runs):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    ##############################################
    ### Plot from the perspective of player1 first ####

    actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)
    state_player1 = pd.read_csv(f'{destination_folder}/player1/state.csv', index_col=0)
    #rename columns 
    colnames = ['run'+str(i) for i in range(n_runs)]
    actions_player1.columns = colnames
    state_player1.columns = colnames
    results = pd.DataFrame(columns=colnames)

    for colname in colnames: 
        str_value = actions_player1[colname].astype(str) + ' | ' + state_player1[colname].astype(str)
        str_value = str_value.str.replace('1', 'D')
        str_value = str_value.str.replace('0', 'C')
        results[colname] = str_value

    #results.to_csv(str(destination_folder+'/player1/action_types_full.csv'))
    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first apisode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    #plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.area(stacked=True, ylabel = '# agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
        xlabel='Iteration', colormap='PiYG_r',
        #color={'C | (C,C)':'lightblue', 'C | (C,D)':'deepskyblue', 'C | (D,C)':'cyan', 'C | (D,D)':'royalblue', 'D | (C,C)':'orange', 'D | (C,D)':'yellow', 'D | (D,C)':'peru', 'D | (D,D)':'dimgrey'}, 
        title='Types of actions over time: \n '+player1_title+' agent (player1) against '+player2_title+' agent')
    plt.savefig(f'{destination_folder}/plots/action_types_area_player1.png', bbox_inches='tight')


    ##############################################
    ### Plot from the perspective of player2 ####

    actions_player2 = pd.read_csv(f'{destination_folder}/player2/action.csv', index_col=0)
    state_player2 = pd.read_csv(f'{destination_folder}/player2/state.csv', index_col=0)
    #rename columns - use colnames form above
    actions_player2.columns = colnames
    state_player2.columns = colnames
    results = pd.DataFrame(columns=colnames)

    for colname in colnames: 
        str_value = actions_player2[colname].astype(str) + ' | ' + state_player2[colname].astype(str)
        str_value = str_value.str.replace('1', 'D')
        str_value = str_value.str.replace('0', 'C')
        results[colname] = str_value

    #results.to_csv(str(destination_folder+'/player2/action_types_full.csv'))
    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:]
    results_counts.dropna(axis=1, how='all', inplace=True)

    #plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.area(stacked=True, ylabel = '# agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
        xlabel='Iteration', colormap='PiYG_r',
        #color={'C | C':'lightblue', 'C | D':'yellow', 'D | C':'brown', 'D | D':'orange'},
        #color={'C | (C,C)':'lightblue', 'C | (C,D)':'yellow', 'D | (D,C)':'brown', 'D | (D,D)':'orange',
        #        'D | (C,C)':'darkblue', 'D | (C,D)':'mustard', 'C | (D,C)':'grey', 'C | (D,D)':'red'}, 
        title='Types of actions over time: \n '+player2_title+' agent (player2) against '+player1_title+' agent')
    plt.savefig(f'{destination_folder}/plots/action_types_area_player2.png', bbox_inches='tight')


def plot_last_20_actions(destination_folder, player1_title, player2_title, n_runs):
    '''deep-dive into the strategy learnt by the end - interpert the last 20 moves
    NOTE this also generates the csv last_20_actions.csv that we can use to visualise a matrix later'''
    ##############################################
    ### Plot from the perspective of player1 first ####

    actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)[-20:]
    state_player1 = pd.read_csv(f'{destination_folder}/player1/state.csv', index_col=0)[-20:]
    #rename columns 
    colnames = ['run'+str(i) for i in range(n_runs)]
    actions_player1.columns = colnames
    state_player1.columns = colnames
    results = pd.DataFrame(columns=colnames)

    for colname in colnames: 
        str_value = actions_player1[colname].astype(str) + ' | ' + state_player1[colname].astype(str)
        str_value = str_value.str.replace('1', 'D')
        str_value = str_value.str.replace('0', 'C')
        results[colname] = str_value

    results.to_csv(str(destination_folder+'/player1/last_20_actions.csv'))
    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first apisode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.bar(stacked=True, ylabel = '# agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
        xlabel='Iteration', colormap='PiYG_r',
        #color={'C | C':'lightblue', 'C | D':'yellow', 'D | C':'brown', 'D | D':'orange'}, 
        title='Last 20 action types: \n '+player1_title+' agent (player1) against '+player2_title+' agent')
    plt.savefig(f'{destination_folder}/plots/last_20_actions_player1.png', bbox_inches='tight')


    ##############################################
    ### Plot from the perspective of player2 ####

    actions_player2 = pd.read_csv(f'{destination_folder}/player2/action.csv', index_col=0)[-20:]
    state_player2 = pd.read_csv(f'{destination_folder}/player2/state.csv', index_col=0)[-20:]
    #rename columns - use colnames form above
    actions_player2.columns=colnames
    state_player2.columns=colnames
    results = pd.DataFrame(columns=colnames)

    for colname in colnames: 
        str_value = actions_player2[colname].astype(str) + ' | ' + state_player2[colname].astype(str)
        str_value = str_value.str.replace('1', 'D')
        str_value = str_value.str.replace('0', 'C')
        results[colname] = str_value

    results.to_csv(str(destination_folder+'/player2/last_20_actions.csv'))
    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:]
    results_counts.dropna(axis=1, how='all', inplace=True)

    plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.bar(stacked=True, ylabel = '# agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
        xlabel='Iteration', colormap='PiYG_r',
        #color={'C | C':'lightblue', 'C | D':'yellow', 'D | C':'brown', 'D | D':'orange'}, 
        title='Last 20 action types: \n '+player2_title+' agent (player2) against '+player1_title+' agent')
    plt.savefig(f'{destination_folder}/plots/last_20_actions_player2.png', bbox_inches='tight')

## NB keep this!!! 
def C_condition(v):
        if v == 'C | (C, C)':
            color = "#28641E"
        elif v =='C | (C, D)':
            color = "#63A336"
        elif v =='C | (D, C)':
            color = "#B0DC82"
        elif v =='C | (D, D)':
            color = "#EBF6DC"
        elif v =='D | (C, C)':
            color = "#FBE6F1"
        elif v =='D | (C, D)':
            color = "#EEAED4"
        elif v =='D | (D, C)':
            color = "#CE4591"
        elif v == 'D | (D, D)':
            color = "#8E0B52"
        else:
            color = "#A9A9A9"
        return 'background-color: %s' % color

action_pairs = False
if action_pairs:         
    reference = pd.DataFrame(['C | (C, C)', 'C | (C, D)', 'C | (D, C)', 'C | (D, D)', 'D | (C, C)', 'D | (C, D)', 'D | (D, C)', 'D | (D, D)', float("NaN")])
    reference = reference.style.applymap(C_condition).set_caption(f"Colour map for action types")
    #dfi.export(reference,"results/reference_color_map_for_action_types.png")


def visualise_last_20_actions_matrix(destination_folder):
    '''explore individual strategies learnt by individual players (not a collection of 100 players) 
    - look at 20 last moves as a vector
    NOTE plot_last_20_actions needs to be run first, to create the last_20_actions csv'''

    results_player1 = pd.read_csv(str(destination_folder+'/player1/last_20_actions.csv'), index_col=0).transpose()
    results_player2 = pd.read_csv(str(destination_folder+'/player2/last_20_actions.csv'), index_col=0).transpose()

    caption = destination_folder.replace('results/', '')
    caption = caption.replace('QL', '')
    results_player1 = results_player1.style.applymap(C_condition).set_caption(f"Player1 from {caption}")
    dfi.export(results_player1,f"{destination_folder}/plots/table_export_player1_last20.png")

    results_player2 = results_player2.style.applymap(C_condition).set_caption(f"Player2 from {caption}")
    dfi.export(results_player2,f"{destination_folder}/plots/table_export_player2_last20.png")

def plot_first_20_actions(destination_folder, player1_title, player2_title, n_runs):
    '''deep-dive into the strategy learnt by the end - interpert the last 20 moves
    NOTE this also generates the csv last_20_actions.csv that we can use to visualise a matrix later'''
    ##############################################
    ### Plot from the perspective of player1 first ####

    actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)[0:20]
    state_player1 = pd.read_csv(f'{destination_folder}/player1/state.csv', index_col=0)[0:20]
    #rename columns 
    colnames = ['run'+str(i) for i in range(n_runs)]
    actions_player1.columns = colnames
    state_player1.columns = colnames
    results = pd.DataFrame(columns=colnames)

    for colname in colnames: 
        str_value = actions_player1[colname].astype(str) + ' | ' + state_player1[colname].astype(str)
        str_value = str_value.str.replace('1', 'D')
        str_value = str_value.str.replace('0', 'C')
        results[colname] = str_value

    results.to_csv(str(destination_folder+'/player1/first_20_actions.csv'))
    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first apisode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.bar(stacked=True, ylabel = '# agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
        xlabel='Iteration', colormap='PiYG_r',
        #color={'C | C':'lightblue', 'C | D':'yellow', 'D | C':'brown', 'D | D':'orange'}, 
        title='First 20 action types: \n '+player1_title+' agent (player1) against '+player2_title+' agent')
    plt.savefig(f'{destination_folder}/plots/first_20_actions_player1.png', bbox_inches='tight')


    ##############################################
    ### Plot from the perspective of player2 ####

    actions_player2 = pd.read_csv(f'{destination_folder}/player2/action.csv', index_col=0)[0:20]
    state_player2 = pd.read_csv(f'{destination_folder}/player2/state.csv', index_col=0)[0:20]
    #rename columns - use colnames form above
    actions_player2.columns=colnames
    state_player2.columns=colnames
    results = pd.DataFrame(columns=colnames)

    for colname in colnames: 
        str_value = actions_player2[colname].astype(str) + ' | ' + state_player2[colname].astype(str)
        str_value = str_value.str.replace('1', 'D')
        str_value = str_value.str.replace('0', 'C')
        results[colname] = str_value

    results.to_csv(str(destination_folder+'/player2/first_20_actions.csv'))
    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:]
    results_counts.dropna(axis=1, how='all', inplace=True)

    plt.figure(figsize=(20, 15), dpi=100)
    results_counts.plot.bar(stacked=True, ylabel = '# agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
        xlabel='Iteration', colormap='PiYG_r',
        #color={'C | C':'lightblue', 'C | D':'yellow', 'D | C':'brown', 'D | D':'orange'}, 
        title='First 20 action types: \n '+player2_title+' agent (player2) against '+player1_title+' agent')
    plt.savefig(f'{destination_folder}/plots/first_20_actions_player2.png', bbox_inches='tight')

def visualise_first_20_actions_matrix(destination_folder):
    '''explore individual strategies learnt by individual players (not a collection of 100 players) 
    - look at 20 first moves as a vector
    NOTE plot_first_20_actions needs to be run first, to create the first_20_actions csv'''

    results_player1 = pd.read_csv(str(destination_folder+'/player1/first_20_actions.csv'), index_col=0).transpose()
    results_player2 = pd.read_csv(str(destination_folder+'/player2/first_20_actions.csv'), index_col=0).transpose()

    caption = destination_folder.replace('results/', '')
    caption = caption.replace('QL', '')
    results_player1 = results_player1.style.applymap(C_condition).set_caption(f"Player1 from {caption}")
    dfi.export(results_player1,f"{destination_folder}/plots/table_export_player1_first20.png")

    results_player2 = results_player2.style.applymap(C_condition).set_caption(f"Player2 from {caption}")
    dfi.export(results_player2,f"{destination_folder}/plots/table_export_player2_first20.png")

def visualise_first_20_actions_matrix_randomorder(destination_folder):
    '''for debugging - explore individual strategies learnt by individual players (not a collection of 100 players) 
    - look at 20 first moves as a vector
    NOTE plot_first_20_actions needs to be run first, to create the first_20_actions csv'''

    results_player1 = pd.read_csv(str(destination_folder+'/player1/first_20_actions.csv'), index_col=0).transpose()
    results_player2 = pd.read_csv(str(destination_folder+'/player2/first_20_actions.csv'), index_col=0).transpose()

    #shuffle the rows to output a matrix in a different order - for checking our clustering problem 
    results_player1 = results_player1.sample(frac=1)
    results_player2 = results_player2.sample(frac=1)

    caption = destination_folder.replace('results_', '')
    caption = caption.replace('QL', '')
    results_player1 = results_player1.style.applymap(C_condition).set_caption(f"Player1 from {caption}; randomly shuffled rows")
    dfi.export(results_player1,f"{destination_folder}/plots/table_export_player1_first20_randomorder.png")
    results_player2 = results_player2.style.applymap(C_condition).set_caption(f"Player2 from {caption}; randomly shuffled rows")
    dfi.export(results_player2,f"{destination_folder}/plots/table_export_player2_first20_randomorder.png")

def plot_one_run_Q_values(Q_values_list, run_idx):
    '''plot pregression of Q-value updates over the 10000 iterations for one example run - separately for each state.
    We plot both actions (C and D) on one plot to copare which action ends up being optimal in every state.'''
    rXs0a0_list = [iteration[0,0] for iteration in Q_values_list[run_idx]] #run x, state0, action1
    rXs0a1_list = [iteration[0,1] for iteration in Q_values_list[run_idx]]
    rXs1a0_list = [iteration[1,0] for iteration in Q_values_list[run_idx]]
    rXs1a1_list = [iteration[1,1] for iteration in Q_values_list[run_idx]]
    rXs2a0_list = [iteration[2,0] for iteration in Q_values_list[run_idx]]
    rXs2a1_list = [iteration[2,1] for iteration in Q_values_list[run_idx]]
    rXs3a0_list = [iteration[3,0] for iteration in Q_values_list[run_idx]]
    rXs3a1_list = [iteration[3,1] for iteration in Q_values_list[run_idx]]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    #fig.suptitle(f'Q-values for all state-action pairs, run{run_idx} \n')
    fig.suptitle(str(f'run{run_idx}'+'\n'))
    ax1.plot(range(len(rXs0a0_list)), rXs0a0_list[:], label=f'Action=C', color='darkgreen')
    ax1.plot(range(len(rXs0a1_list)), rXs0a1_list[:], label=f'Action=D', color='purple')
    ax1.set_title('State=(C,C)')
    ax2.plot(range(len(rXs1a0_list)), rXs1a0_list[:], label=f'Action=C', color='darkgreen')
    ax2.plot(range(len(rXs1a1_list)), rXs1a1_list[:], label=f'Action=D', color='purple')
    ax2.set_title('State=(C,D)')
    ax3.plot(range(len(rXs2a0_list)), rXs2a0_list[:], label=f'Action=C', color='darkgreen')
    ax3.plot(range(len(rXs2a1_list)), rXs2a1_list[:], label=f'Action=D', color='purple')
    ax3.set_title('State=(D,C)')
    ax4.plot(range(len(rXs3a0_list)), rXs3a0_list[:], label=f'Action=C', color='darkgreen')
    ax4.plot(range(len(rXs3a1_list)), rXs3a1_list[:], label=f'Action=D', color='purple')
    ax4.set_title('State=(D,D)')
    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    ax1.set(ylabel='Q-value')
    #max_Q_value = max(max(rXs0a0_list), max(rXs0a1_list), max(rXs1a0_list), max(rXs1a1_list)) #max q-value on this run
    #plt.ylim(0, int(max_Q_value+5))
    #set y axis limit based on the claculation of a max possible return in this game and the discount factor: r=4, gamma=0.9, max return = r*(1/1-gamma)
    
    #if game_title == 'IPD': 
    #    plt.ylim(0, 40+5) 
    #elif game_title == 'VOLUNTEER':
    #    plt.ylim(0, 50+5)
    #elif game_title == 'STAGHUNT':
    #    plt.ylim(0, 50+5)
    #then export the interactive outupt / single plot as pdf/html to store the results compactly 


def plot_diff_one_run_Q_values(Q_values_list, run_idx):
    '''plot pregression of Q-value updates over the 10000 iterations for one example run - separately for each state.
    We plot the difference between Q(C) and Q(D) on one plot to compare which action ends up being optimal in every state.
    values >1 = Cooperation is preferred 
    values <1 = Defection is preferred '''
    rXs0a0_list = [iteration[0,0] for iteration in Q_values_list[run_idx]] #run x, state0, action1
    rXs0a1_list = [iteration[0,1] for iteration in Q_values_list[run_idx]]
    rxs0_list = np.array(rXs0a0_list) - np.array(rXs0a1_list)
    rXs1a0_list = [iteration[1,0] for iteration in Q_values_list[run_idx]]
    rXs1a1_list = [iteration[1,1] for iteration in Q_values_list[run_idx]]
    rxs1_list = np.array(rXs1a0_list) - np.array(rXs1a1_list)
    rXs2a0_list = [iteration[2,0] for iteration in Q_values_list[run_idx]]
    rXs2a1_list = [iteration[2,1] for iteration in Q_values_list[run_idx]]
    rxs2_list = np.array(rXs2a0_list) - np.array(rXs2a1_list)
    rXs3a0_list = [iteration[3,0] for iteration in Q_values_list[run_idx]]
    rXs3a1_list = [iteration[3,1] for iteration in Q_values_list[run_idx]]
    rxs3_list = np.array(rXs3a0_list) - np.array(rXs3a1_list)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    #fig.suptitle(f'Q-values for all state-action pairs, run{run_idx} \n')
    fig.suptitle(str(f'run{run_idx}'+'\n'))
    ax1.plot(range(len(rxs0_list)), np.zeros(len(rxs0_list)), label=f'reference: Q(C) = Q(D)', color='black')
    ax1.plot(range(len(rxs0_list)), rxs0_list[:], label=f'Preference of Action=C over Action=D', color='red')
    ax1.set_title('State=(C,C)')
    ax2.plot(range(len(rxs0_list)), np.zeros(len(rxs0_list)), label=f'reference: Q(C) = Q(D)', color='black')
    ax2.plot(range(len(rxs1_list)), rxs1_list[:], label=f'Preference of Action=C over Action=D', color='red')
    ax2.set_title('State=(C,D)')
    ax3.plot(range(len(rxs0_list)), np.zeros(len(rxs0_list)), label=f'reference: Q(C) = Q(D)', color='black')
    ax3.plot(range(len(rxs2_list)), rxs2_list[:], label=f'Preference of Action=C over Action=D', color='red')
    ax3.set_title('State=(D,C)')
    ax4.plot(range(len(rxs0_list)), np.zeros(len(rxs0_list)), label=f'reference : Q(C) = Q(D)', color='black')
    ax4.plot(range(len(rxs3_list)), rxs3_list[:], label=f'Preference of Action=C over Action=D', color='red')
    ax4.set_title('State=(D,D)')
    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    ax1.set(ylabel='Q(C) - Q(D)')
    #max_Q_value = max(max(rXs0a0_list), max(rXs0a1_list), max(rXs1a0_list), max(rXs1a1_list)) #max q-value on this run
    #plt.ylim(0, int(max_Q_value+5))

def plot_Q_values(destination_folder, n_runs, which, fewer_than_nruns = False):
    if fewer_than_nruns == True:
        n_runs = 10 

    #plot for player1
    if which == 'local':
        history_Qvalues_local_player1 = np.load(f'{destination_folder}/Q_VALUES_local_player1_list.npy', allow_pickle=True)
        print('Printing local Q-values, player1:')
        for run_idx in range(n_runs):
            plot_one_run_Q_values(Q_values_list = history_Qvalues_local_player1, run_idx = run_idx)
#            plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_local_player1, run_idx = run_idx)
    elif which == 'target':
        try: 
            print('Printing target Q-values, player1:')
            history_Qvalues_target_player1 = np.load(f'{destination_folder}/Q_VALUES_target_player1_list.npy', allow_pickle=True)
            for run_idx in range(n_runs):
                plot_one_run_Q_values(Q_values_list = history_Qvalues_target_player1, run_idx = run_idx)
#                plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_target_player1, run_idx = run_idx)
        except:
            print('Unable to locate target Q-values, player1')

    #plot for player2
    if which == 'local':
        try:
            history_Qvalues_local_player2 = np.load(f'{destination_folder}/Q_VALUES_local_player2_list.npy', allow_pickle=True)
            print('Printing local Q-values, player2:')
            for run_idx in range(n_runs):
                plot_one_run_Q_values(Q_values_list = history_Qvalues_local_player2, run_idx = run_idx)
#                plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_local_player2, run_idx = run_idx)
        except: 
            print('Unable to locate local Q-values, player2')

    elif which == 'target':
        try: 
            print('Printing target Q-values, player2:')
            history_Qvalues_target_player2 = np.load(f'{destination_folder}/Q_VALUES_target_player2_list.npy', allow_pickle=True)
            for run_idx in range(n_runs):
                plot_one_run_Q_values(Q_values_list = history_Qvalues_target_player2, run_idx = run_idx)
#                plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_target_player2, run_idx = run_idx)
        except:
            print('Unable to locate target Q-values, player2')

def plot_diff_Q_values(destination_folder, n_runs, which, fewer_than_nruns = False):
    if fewer_than_nruns == True:
        n_runs = 10 

    #plot for player1
    if which == 'local':
        history_Qvalues_local_player1 = np.load(f'{destination_folder}/Q_VALUES_local_player1_list.npy', allow_pickle=True)
        print('Printing local Q-values, player1:')
        for run_idx in range(n_runs):
#            plot_one_run_Q_values(Q_values_list = history_Qvalues_local_player1, run_idx = run_idx)
            plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_local_player1, run_idx = run_idx)
    elif which == 'target':
        try: 
            print('Printing target Q-values, player1:')
            history_Qvalues_target_player1 = np.load(f'{destination_folder}/Q_VALUES_target_player1_list.npy', allow_pickle=True)
            for run_idx in range(n_runs):
#                plot_one_run_Q_values(Q_values_list = history_Qvalues_target_player1, run_idx = run_idx)
                plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_target_player1, run_idx = run_idx)
        except:
            print('Unable to locate target Q-values, player1')

    #plot for player2
    if which == 'local':
        try:
            history_Qvalues_local_player2 = np.load(f'{destination_folder}/Q_VALUES_local_player2_list.npy', allow_pickle=True)
            print('Printing local Q-values, player2:')
            for run_idx in range(n_runs):
#                plot_one_run_Q_values(Q_values_list = history_Qvalues_local_player2, run_idx = run_idx)
                plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_local_player2, run_idx = run_idx)
        except: 
            print('Unable to locate local Q-values, player2')

    elif which == 'target':
        try: 
            print('Printing target Q-values, player2:')
            history_Qvalues_target_player2 = np.load(f'{destination_folder}/Q_VALUES_target_player2_list.npy', allow_pickle=True)
            for run_idx in range(n_runs):
#                plot_one_run_Q_values(Q_values_list = history_Qvalues_target_player2, run_idx = run_idx)
                plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_target_player2, run_idx = run_idx)
        except:
            print('Unable to locate target Q-values, player2')




def plot_action_pairs(destination_folder, player1_title, player2_title, n_runs, option=None):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    if False: #if os.path.exists(str(destination_folder+'/action_pairs.csv')):
        results = pd.read_csv(str(destination_folder+'/action_pairs.csv'), index_col=0)

    else: #if action_pairs.csv does not yet exist 
        actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)
        actions_player2 = pd.read_csv(f'{destination_folder}/player2/action.csv', index_col=0)

        #rename columns 
        colnames = ['run'+str(i) for i in range(n_runs)]
        actions_player1.columns = colnames
        actions_player2.columns = colnames

        results = pd.DataFrame(columns=colnames)

        for colname in colnames: #loop over every run
            str_value = actions_player1[colname].astype(str) + ', ' + actions_player2[colname].astype(str)
            str_value = str_value.str.replace('1', 'D')
            str_value = str_value.str.replace('0', 'C')
            results[colname] = str_value

        results.to_csv(str(destination_folder+'/action_pairs.csv'))

    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first apisode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    #plt.figure(figsize=(20, 15), dpi=100)
    plt.figure(dpi=80, figsize=(5, 4))
    plt.rcParams.update({'font.size':20})
    results_counts.plot.area(stacked=True, ylabel = '# action pairs observed', #rot=45,
        xlabel='Iteration', #colormap='PiYG_r',
        color={'C, C':'#28641E', 'C, D':'#B0DC82', 'D, C':'#EEAED4', 'D, D':'#8E0B52'}, linewidth=0.05, alpha=0.9,
        #color={'C, C':'royalblue', 'C, D':'lightblue', 'D, C':'yellow', 'D, D':'orange'},
        title=str(player1_title.replace('Ethics','').replace('_','-') +' vs '+player2_title.replace('Ethics','').replace('_','-'))) #Pairs of simultaneous actions over time: \n '+

    #plt.savefig(f'{destination_folder}/plots/action_pairs.png', bbox_inches='tight')
    if not os.path.isdir('results/outcome_plots/actions'):
            os.makedirs('results/outcome_plots/actions')

    if not option: #if plotting main results
        pair = destination_folder.split('/')[1]

    else: #if plotting extra parameter search for beta in QLVM 
        pair = destination_folder
        #pair += str(option)
    
    plt.savefig(f'results/outcome_plots/actions/pairs_{pair}.pdf', bbox_inches='tight')

def plot_action_pairs_reduced(destination_folder, player1_title, player2_title, n_runs, option=None):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    if False: #if os.path.exists(str(destination_folder+'/action_pairs.csv')):
        results = pd.read_csv(str(destination_folder+'/action_pairs.csv'), index_col=0)

    else: #if action_pairs.csv does not yet exist 
        actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)
        actions_player2 = pd.read_csv(f'{destination_folder}/player2/action.csv', index_col=0)

        #rename columns 
        colnames = ['run'+str(i) for i in range(n_runs)]
        actions_player1.columns = colnames
        actions_player2.columns = colnames

        results = pd.DataFrame(columns=colnames)

        for colname in colnames: #loop over every run
            str_value = actions_player1[colname].astype(str) + ', ' + actions_player2[colname].astype(str)
            str_value = str_value.str.replace('1', 'D')
            str_value = str_value.str.replace('0', 'C')
            results[colname] = str_value

        results.to_csv(str(destination_folder+'/action_pairs.csv'))

    #keep every 100th row 
    new_results = results[results.index % 20 == 0]

    results_counts = new_results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first episode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    #plt.figure(figsize=(20, 15), dpi=100)
    plt.figure(dpi=80, figsize=(5, 4))
    plt.rcParams.update({'font.size':20})
    results_counts.plot.area(stacked=True, ylabel = '# action pairs observed', #rot=45,
        xlabel='Iteration', #colormap='PiYG_r',
        color={'C, C':'#28641E', 'C, D':'#B0DC82', 'D, C':'#EEAED4', 'D, D':'#8E0B52'}, linewidth=0.05, alpha=0.9,
        #color={'C, C':'royalblue', 'C, D':'lightblue', 'D, C':'yellow', 'D, D':'orange'},
        title=str(player1_title.replace('Ethics','').replace('_','-') +' vs '+player2_title.replace('Ethics','').replace('_','-'))) #Pairs of simultaneous actions over time: \n '+

    #plt.savefig(f'{destination_folder}/plots/action_pairs.png', bbox_inches='tight')
    if not os.path.isdir('results/outcome_plots/actions'):
            os.makedirs('results/outcome_plots/actions')

    if not option: #if plotting main results
        pair = destination_folder.split('/')[1]

    else: #if plotting extra parameter search for beta in QLVM 
        pair = destination_folder
        #pair += str(option)
    
    plt.savefig(f'results/outcome_plots/actions/pairs_{pair}.pdf', bbox_inches='tight')



def plot_one_run_Loss(LOSS_list, run_idx):
    '''plot pregression of loss values over the 10000 iterations for one example run.'''

    #plt.plot(LOSS_list[run_idx], label='loss for DQN Player')#, ylab = 'Loss', xlab = 'Iteration')
    plt.figure(figsize=(7,3))
    plt.plot(LOSS_list, label='loss for DQN Player', linewidth=0.3)#, ylab = 'Loss', xlab = 'Iteration')
    plt.title("Loss over time for DQN player")
    #plt.ylim(0, max(LOSS_list[run_idx]+1))
    plt.ylim(0, max(LOSS_list)+1)

    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    plt.show()

    #then export the interactive outupt / single plot as pdf/html to store the results compactly 

def plot_one_run_Loss_population(LOSS_list, run_idx, player_idx, selection=True):
    '''plot pregression of loss values over the 10000 iterations for one example run.'''
    LOSS_list = np.array(LOSS_list)
    #plt.plot(LOSS_list[run_idx], label='loss for DQN Player')#, ylab = 'Loss', xlab = 'Iteration')
    plt.figure(figsize=(7,3))
 #    plt.plot(LOSS_list, label='loss for DQN Player', linewidth=0.3)#, ylab = 'Loss', xlab = 'Iteration')
    plt.plot(LOSS_list[~np.isnan(LOSS_list)], label='loss for DQN Player', linewidth=0.1)#, ylab = 'Loss', xlab = 'Iteration')
    if selection:
        plt.title(f'Selection Loss over time for Player{player_idx}, run{run_idx}')
    else:    
        plt.title(f'Dilemma Loss over time for Player{player_idx}, run{run_idx}')
    #plt.ylim(0, max(LOSS_list[run_idx]+1))
    plt.ylim(min(LOSS_list[~np.isnan(LOSS_list)])-1, max(LOSS_list[~np.isnan(LOSS_list)])+1)

    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    plt.show()

    #then export the interactive outupt / single plot as pdf/html to store the results compactly 

def plot_Loss(destination_folder, n_runs, player_x, fewer_than_nruns = False):
    if fewer_than_nruns == True:
        n_runs = 10 

    #plot for player_x
    try: 
        LOSS_player_x = np.load(f'{destination_folder}/LOSS_player{player_x}_list.npy', allow_pickle=True)
        print(f'Printing Loss, player{player_x}:')

        #none_list = {} 
        for run_idx in range(n_runs):
            #none_list[run_idx] = np.isnan(LOSS_player1[run_idx])
            loss_list = [i if i else float(0) for i in LOSS_player_x[run_idx]] #replace None with 0 
            loss_list = [float(i) if i!='<0.0001' else float(0) for i in loss_list] #replace all str '<0.001' with 0
            plot_one_run_Loss(LOSS_list = loss_list, run_idx = run_idx)
    except:
        print(f'Unable to locate Loss, player{player_x}')


def save_last_20_actions_matrix(destination_folder):
    '''explore individual strategies learnt by individual players (not a collection of 100 players) 
    - look at 20 last moves as a vector
    NOTE plot_last_20_actions needs to be run first, to create the last_20_actions csv'''

    results_player1 = pd.read_csv(str(destination_folder+'/player1/last_20_actions.csv'), index_col=0).transpose()
    results_player2 = pd.read_csv(str(destination_folder+'/player2/last_20_actions.csv'), index_col=0).transpose()

    #caption = destination_folder.replace('results/', '')
    #caption = caption.replace('QL', '')
    #results_player1 = results_player1.style.applymap(C_condition).set_caption(f"Player1 from {caption}")
    results_player1.to_csv(f"{destination_folder}/plots/table_export_player1_last20.csv")

    #results_player2 = results_player2.style.applymap(C_condition).set_caption(f"Player2 from {caption}")
    results_player2.to_csv(f"{destination_folder}/plots/table_export_player2_last20.csv")


#######################
#### population_plots ####
#######################


def get_titles_for_population(destination_folder, print_titles=False):

    titles = [] 
    temp_titles = destination_folder.split('__')[0]

    for item in temp_titles.split('_'): 
        items = item.split('x')
        if print_titles: 
            print(items)
        for n_repeats in range(int(items[0])):
            if items[1] in ['AC', 'AD', 'TFT', 'Random', 'CD', 'GEa', 'GEb', 'GEc', 'GEd', 'GEe']:
                titles.append(str(items[1]))
            else: 
                titles.append(str('QL'+items[1]))
    
    #TO DO sort out VEe, VEk --> VE_E, VE_k !! 
    titles_new = []
    for title in titles: 
        if 'VE' in title: 
            titles_new.append(title[0:4]+'_'+title[4:])
        else:
            titles_new.append(title)

    return titles_new

def reformat_reward_for_population(destination_folder, n_runs, population_size, long_titles, episodes=True):
    '''create dfs for each moral player type, cregardless of whether they acted as player1 or player2'''
    colnames = ['run'+str(i+1) for i in range(n_runs)]

    if episodes:
        num_iter_new = num_iter * population_size
    else: 
        num_iter_new = num_iter

    for idx in range(population_size): #initialise the results dataframes for each player idx
        globals()[f'R_game_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))
        globals()[f'Rcumul_game_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))

        globals()[f'R_intr_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))
        globals()[f'Rcumul_intr_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))
    
    if False: 
        R_collective = pd.DataFrame(columns=colnames)
        Rcumul_collective = pd.DataFrame(columns=colnames)
        R_gini = pd.DataFrame(columns=colnames)
        Rcumul_gini = pd.DataFrame(columns=colnames)
        R_min = pd.DataFrame(columns=colnames)
        Rcumul_min = pd.DataFrame(columns=colnames)
    

    for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv')
        run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'reward_game_player1', 'reward_game_player2', 'reward_intrinsic_player1', 'reward_intrinsic_player2']]
        run_df['title_p1'].fillna('Selfish', inplace=True)
        run_df['title_p2'].fillna('Selfish', inplace=True)

        #create for player1 & player2 for each index 
        for idx in range(population_size): 
            ##################################
            #### game reward ####
            ##################################
            globals()[f'R_game_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['reward_game_player1'].astype(int)
            globals()[f'R_game_player2_idx{idx}'] = run_df[run_df['idx_p2']==idx]['reward_game_player2'].astype(int)
            globals()[f'R_game_run_idx{idx}'] = globals()[f'R_game_player1_idx{idx}'].append(globals()[f'R_game_player2_idx{idx}'])
            globals()[f'R_game_run_idx{idx}'] = globals()[f'R_game_run_idx{idx}'].sort_index() #sort index, conveer to int  
            globals()[f'R_game_idx{idx}'][run_name] = globals()[f'R_game_run_idx{idx}'] ##insert result for this particular run into the overall df
            globals()[f'Rcumul_game_idx{idx}'][run_name] = np.cumsum(globals()[f'R_game_run_idx{idx}'])

            ##################################
            #### intrinsic reward ####
            ##################################
            globals()[f'R_intr_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['reward_intrinsic_player1']
            globals()[f'R_intr_player2_idx{idx}'] = run_df[run_df['idx_p2']==idx]['reward_intrinsic_player2']
            globals()[f'R_intr_run_idx{idx}'] = globals()[f'R_intr_player1_idx{idx}'].append(globals()[f'R_intr_player2_idx{idx}'])
            globals()[f'R_intr_run_idx{idx}'] = globals()[f'R_intr_run_idx{idx}'].sort_index() #sort index, conveer to int  
            globals()[f'R_intr_idx{idx}'][run_name] = globals()[f'R_intr_run_idx{idx}'] ##insert result for this particular run into the overall df
            globals()[f'Rcumul_intr_idx{idx}'][run_name] = np.cumsum(globals()[f'R_intr_run_idx{idx}'])

        if False: 
            ##################################
            #### collective, gini & min reward ####
            ##################################
            R_collective_run = run_df['reward_collective'] #not filtering by player idx any more
            R_collective[run_name] = R_collective_run.sort_index()
            Rcumul_collective[run_name] = np.cumsum(R_collective_run)

            R_gini_run = run_df['reward_gini'] #not filtering by player idx any more
            R_gini[run_name] = R_gini_run.sort_index()
            Rcumul_gini[run_name] = np.cumsum(R_gini_run)

            R_min_run = run_df['reward_min'] #not filtering by player idx any more
            R_min[run_name] = R_min_run.sort_index()
            Rcumul_min[run_name] = np.cumsum(R_min_run)

    #save the reformatted data
    if not os.path.isdir(f'{destination_folder}/player_rewards'):
            os.makedirs(f'{destination_folder}/player_rewards')

    print('save results...')
    for idx in range(population_size): #save results
        idx_title = long_titles[idx]

        globals()[f'R_game_idx{idx}'].to_csv(f'{destination_folder}/player_rewards/R_game_{idx_title}_{idx}.csv')
        globals()[f'Rcumul_game_idx{idx}'].to_csv(f'{destination_folder}/player_rewards/Rcumul_game_{idx_title}_{idx}.csv')

        globals()[f'R_intr_idx{idx}'].to_csv(f'{destination_folder}/player_rewards/R_intr_{idx_title}_{idx}.csv')
        globals()[f'Rcumul_intr_idx{idx}'].to_csv(f'{destination_folder}/player_rewards/Rcumul_intr_{idx_title}_{idx}.csv')

    if False: 
        if not os.path.isdir(f'{destination_folder}/player_outcomes'):
            os.makedirs(f'{destination_folder}/player_outcomes')

        R_collective.to_csv(f'{destination_folder}/player_outcomes/R_collective.csv')
        Rcumul_collective.to_csv(f'{destination_folder}/player_outcomes/Rcumul_collective.csv')
        R_gini.to_csv(f'{destination_folder}/player_outcomes/R_gini.csv')
        Rcumul_gini.to_csv(f'{destination_folder}/player_outcomes/Rcumul_gini.csv')
        R_min.to_csv(f'{destination_folder}/player_outcomes/R_min.csv')
        Rcumul_min.to_csv(f'{destination_folder}/player_outcomes/Rcumul_min.csv')

def plot_results_for_population(destination_folder, titles, n_runs, game_title): #TO DO !!! 
    '''plot reward over time - cumulative and non-cumulative; game reward and intrinsic and others'''
    if not os.path.isdir(str(destination_folder)+'/plots'):
        os.makedirs(str(destination_folder)+'/plots')

    if not os.path.isdir(str(destination_folder)+'/plots/outcome_plots'):
        os.makedirs(str(destination_folder)+'/plots/outcome_plots')

    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')
    else: 
        for t in titles_short: 
            temp += (t+',')
        population_list = temp.strip(',') 

    ##################################
    #### cumulative - game reward ####
    ##################################
    #plt.figure(dpi=80) #figsize=(10, 6) 
    plt.figure(dpi=80, figsize=(5, 4))
    plt.rcParams.update({'font.size':20})

    #plot for each player type: 
    for idx in range(len(titles)):
        title = title_mapping[titles[idx]]

        my_df_idx = pd.read_csv(f'{destination_folder}/player_rewards/Rcumul_game_{title}_{idx}.csv', index_col=0).sort_index()
        globals()[f'means_idx{idx}'] = my_df_idx.mean(axis=1)
        globals()[f'sds_idx{idx}'] = my_df_idx.std(axis=1)
        #ci = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
        globals()[f'ci_idx{idx}'] = 1.96 * globals()[f'sds_idx{idx}']/np.sqrt(n_runs)


        plt.plot(my_df_idx.index[:], globals()[f'means_idx{idx}'][:], label=f'player{idx} - {title}', linewidth=0.5) #, color='blue'
        plt.fill_between(my_df_idx.index[:], globals()[f'means_idx{idx}']-globals()[f'ci_idx{idx}'], globals()[f'means_idx{idx}']+globals()[f'ci_idx{idx}'], facecolor='#95d0fc', alpha=0.3) #0.7
    
    plt.title('Cumulative Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+f'Population {population_list}')
    plt.ylabel('Cumulative Game reward')
    plt.xlabel('Iteration')
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(2.0)
    plt.savefig(f'{destination_folder}/plots/reward_cumulative_Game.png', bbox_inches='tight')

    ######################################
    #### non-cumulative - game reward ####
    ######################################
    plt.figure(dpi=80) #figsize=(10, 6) 

    #plot for each player type: 
    for idx in range(len(titles)):
        title = title_mapping[titles[idx]]

        my_df_idx = pd.read_csv(f'{destination_folder}/player_rewards/R_game_{title}_{idx}.csv', index_col=0).sort_index()
        globals()[f'means_idx{idx}'] = my_df_idx.mean(axis=1)
        globals()[f'sds_idx{idx}'] = my_df_idx.std(axis=1)
        #ci = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
        globals()[f'ci_idx{idx}'] = 1.96 * globals()[f'sds_idx{idx}']/np.sqrt(n_runs)


        plt.plot(my_df_idx.index[:], globals()[f'means_idx{idx}'][:], label=f'player{idx} - {title}', lw=0.5) #, color='blue'
        plt.fill_between(my_df_idx.index[:], globals()[f'means_idx{idx}']-globals()[f'ci_idx{idx}'], globals()[f'means_idx{idx}']+globals()[f'ci_idx{idx}'], facecolor='#95d0fc', alpha=0.3) #0.7
        plt.title('Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+f'Population {population_list}')
    #plt.gca().set_ylim([1, 4]) #[0,5]
    if game_title=='IPD':
        plt.gca().set_ylim([1, 4])
    elif game_title=='VOLUNTEER':
        plt.gca().set_ylim([1, 5])
    elif game_title=='STAGHUNT':  
       plt.gca().set_ylim([1, 5])  
    plt.ylabel('Game reward')
    plt.xlabel('Iteration')
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)
    plt.savefig(f'{destination_folder}/plots/reward_Game.png', bbox_inches='tight')

    if False: #do not plot at the moment
        plt.figure(figsize=(10, 6), dpi=80)
        plt.plot(my_df_player1.index[:], means_player1[:], lw=0.5, label=f'player1 - {player1_title}', color='blue')
        plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
        plt.title('Game Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+player1_title)
        plt.gca().set_ylim([0, 5])
        plt.ylabel('Game reward')
        plt.xlabel('Iteration')
        plt.legend(loc='upper left')
        plt.savefig(f'{destination_folder}/plots/reward_Game_player1.png')


        plt.figure(figsize=(10, 6), dpi=80)
        plt.plot(my_df_player2.index[:], means_player2[:], lw=0.5, label=f'player2 - {player2_title}', color='orange')
        plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
        plt.title(r'Game Reward (Mean over '+str(n_runs)+r' runs $\pm$ SD), '+player2_title)
        plt.gca().set_ylim([0, 5])
        plt.ylabel('Game reward')
        plt.xlabel('Iteration')
        plt.legend(loc='upper left')
        plt.savefig(f'{destination_folder}/plots/reward_Game_player2.png')


if False: 
    ##################################
    #### cumulative - intrinsic reward ###
    ##################################
    my_df_player1 = pd.read_csv(f'{destination_folder}/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
    means_player1 = my_df_player1.mean(axis=1)
    sds_player1 = my_df_player1.std(axis=1)

    my_df_player2 = pd.read_csv(f'{destination_folder}/player2/df_cumulative_reward_intrinsic.csv', index_col=0)
    means_player2 = my_df_player2.mean(axis=1)
    sds_player2 = my_df_player2.std(axis=1)

    #plot player1 and player2 separately
    #only plot player1 intrinsic reward if they are a QL player 
    if 'QL' in player1_title:
        if 'Selfish' not in player1_title: 
            plt.figure(dpi=80) #figsize=(10, 6)
            plt.plot(my_df_player1.index[:], means_player1[:], label=f'player1 - {player1_title}', color='blue')
            #plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
            plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
            #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            #plt.title(r'Cumulative Intrinsic Reward (Mean over '+str(n_runs)+r' runs $\pm$ SD), '+player1_title+' vs '+player2_title)
            plt.title('Cumulative Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title)
            #plt.title(r'Cumulative Intrinsic Reward (Mean over '+str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title)
            plt.ylabel('Cumulative Intrinsic reward')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/reward_cumulative_Intrinsic_player1.png', bbox_inches='tight')

    #only plot player2 intrinsic reward if they are a QL player 
    if 'QL' in player2_title:
        if 'Selfish' not in player2_title: 
            plt.figure(dpi=80) #figsize=(10, 6)
            plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
            plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            plt.title('Cumulative Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player2_title)
            #plt.title(r'Cumulative Intrinsic Reward (Mean over '+str(n_runs)+r' runs $\pm$ SD), '+'\n'+player2_title)
            plt.ylabel('Cumulative Intrinsic reward')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/reward_cumulative_Intrinsic_player2.png', bbox_inches='tight')


    ######################################
    #### non-cumulative - intrinsic reward ###
    ######################################
    #only plot player1 intrinsic reward if they are a QL player 
    if 'QL' in player1_title:
        if 'Selfish' not in player1_title: 
            my_df_player1 = pd.read_csv(f'{destination_folder}/player1/df_reward_intrinsic.csv', index_col=0)
            means_player1 = my_df_player1.mean(axis=1)
            sds_player1 = my_df_player1.std(axis=1)

            plt.figure(dpi=80) #figsize=(10, 6)
            plt.plot(my_df_player1.index[:], means_player1[:], lw=0.5, label=f'player1 - {player1_title}', color='blue')
            #plt.plot(my_df_player2.index[:], means_player2[:], lw=0.5, label=f'player2 - {player2_title}', color='orange')
            plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
            #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            plt.title('Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title)
            #plt.title(r'Intrinsic Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title)
            if game_title=='IPD':
                plt.gca().set_ylim([-5, 6])
            elif game_title=='VOLUNTEER':
                plt.gca().set_ylim([-5, 8])
            elif game_title=='STAGHUNT':  
                plt.gca().set_ylim([-5, 10])  
            plt.ylabel('Intrinsic reward')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/reward_Intrinsic_player1.png', bbox_inches='tight')

    if 'QL' in player2_title:
        if 'Selfish' not in player2_title: 
            my_df_player2 = pd.read_csv(f'{destination_folder}/player2/df_reward_intrinsic.csv', index_col=0)
            means_player2 = my_df_player2.mean(axis=1)
            sds_player2 = my_df_player2.std(axis=1)

            plt.figure(dpi=80) #figsize=(10, 6)
            plt.plot(my_df_player2.index[:], means_player2[:], lw=0.5, label=f'player2 - {player2_title}', color='orange')
            plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            plt.title('Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player2_title)
            #plt.title(r'Intrinsic Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player2_title)
            if game_title=='IPD':
                plt.gca().set_ylim([-5, 6])
            elif game_title=='VOLUNTEER':
                plt.gca().set_ylim([-5, 8])
            elif game_title=='STAGHUNT':  
                plt.gca().set_ylim([-5, 10])
            plt.ylabel('Intrinsic reward')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/reward_Intrinsic_player2.png', bbox_inches='tight')

    if False: #do not plot at the moment 
        ##################################
        #### cumulative - collective game reward ####
        ##################################
        my_df = pd.read_csv(f'{destination_folder}/df_cumulative_reward_collective.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)


        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
    #    plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        plt.title('Cumulative Collective Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title+' vs '+player2_title)
        #plt.title(r'Cumulative Collective Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        plt.ylabel('Cumulative Collective reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_cumulative_Collective.png', bbox_inches='tight')

        ######################################
        #### non-cumulative - collective game reward ####
        ######################################
        #TO DO 
        my_df = pd.read_csv(f'{destination_folder}/df_reward_collective.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
    #    plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        plt.title('Collective Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+player1_title+' vs '+player2_title)
        #plt.title(r'Collective Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        if game_title=='IPD':
            plt.gca().set_ylim([4, 6])
        elif game_title=='VOLUNTEER':
            plt.gca().set_ylim([2, 8])
        elif game_title=='STAGHUNT': 
            plt.gca().set_ylim([4, 10])
        plt.ylabel('Collective reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_Collective.png', bbox_inches='tight')


        ##################################
        #### cumulative - gini game reward ####
        ##################################
        my_df = pd.read_csv(f'{destination_folder}/df_cumulative_reward_gini.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        #plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        #plt.title('Cumulative Gini Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+'runs), '+'\n'+player1_title+' vs '+player2_title)
        plt.title(r'Cumulative Gini Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        plt.ylabel('Cumulative Gini reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_cumulative_Gini.png', bbox_inches='tight')

        ######################################
        #### non-cumulative - gini game reward ####
        ######################################
        #TO DO 
        my_df = pd.read_csv(f'{destination_folder}/df_reward_gini.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        #plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        #plt.title('Gini Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+'runs), '+'\n'+player1_title+' vs '+player2_title)
        plt.title(r'Gini Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        plt.gca().set_ylim([0, 1])
        plt.ylabel('Gini reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_Gini.png', bbox_inches='tight')




        ##################################
        #### cumulative - min game reward ####
        ##################################
        my_df = pd.read_csv(f'{destination_folder}/df_cumulative_reward_min.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        #plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        #plt.title('Cumulative Min Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+'runs), '+'\n'+player1_title+' vs '+player2_title)
        plt.title(r'Cumulative Min Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        plt.ylabel('Cumulative Min reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_cumulative_Min.png', bbox_inches='tight')

        ######################################
        #### non-cumulative - min game reward ####
        ######################################
        #TO DO 
        my_df = pd.read_csv(f'{destination_folder}/df_reward_min.csv', index_col=0)
        means = my_df.mean(axis=1)
        sds = my_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
        plt.fill_between(my_df.index[:], means-ci, means+ci, facecolor='#bf92e4', alpha=0.7)
        #plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
        #plt.title('Min Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+'runs), '+'\n'+player1_title+' vs '+player2_title)
        plt.title(r'Min Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
        if game_title=='IPD':
            plt.gca().set_ylim([1, 3])
        elif game_title=='VOLUNTEER':
            plt.gca().set_ylim([1, 4])
        elif game_title=='STAGHUNT': 
            plt.gca().set_ylim([1, 5]) 
        plt.ylabel('Min reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/outcome_plots/reward_Min.png', bbox_inches='tight')



def reformat_a_s_for_population(destination_folder, n_runs, population_size):
    '''create dfs for each moral player type, cregardless of whether they acted as player1 or player2'''
    colnames = ['run'+str(i+1) for i in range(n_runs)]
    for idx in range(population_size): #initialise the results dataframes for each player idx
        globals()[f'results_idx{idx}'] = pd.DataFrame(columns=colnames)
    

    for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv')
        run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
        run_df['title_p1'].fillna('Selfish', inplace=True)
        run_df['title_p2'].fillna('Selfish', inplace=True)

        #create for player1 & player2 for each index 
        for idx in range(population_size): 
            globals()[f'results_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['action_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==idx]['state_player1'].astype(str)
            globals()[f'results_player2_idx{idx}'] = run_df[run_df['idx_p2']==idx]['action_player2'].astype(str) + ' | ' + run_df[run_df['idx_p2']==idx]['state_player2'].astype(str)           
            globals()[f'results_run_idx{idx}'] = globals()[f'results_player1_idx{idx}'].append(globals()[f'results_player2_idx{idx}'])

            globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('1', 'D')
            globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('0', 'C')
            globals()[f'results_idx{idx}'][run_name] = globals()[f'results_run_idx{idx}'] #insert result for this particular run into the overall df 
    
    #save the reformatted data
    if not os.path.isdir(f'{destination_folder}/player_action_types'):
            os.makedirs(f'{destination_folder}/player_action_types')
    
    for idx in range(population_size): #save results
        idx_title = run_df.loc[run_df['idx_p1']==idx]['title_p1']
        idx_title = idx_title[idx_title.first_valid_index()]
        globals()[f'results_idx{idx}'].to_csv(f'{destination_folder}/player_action_types/{idx_title}_{idx}.csv')
    

def reformat_sel_s_for_population(destination_folder, n_runs, population_size):
    '''create dfs for each moral player type, cregardless of whether they acted as player1 or player2'''
    colnames = ['run'+str(i+1) for i in range(n_runs)]
    for idx in range(population_size): #initialise the results dataframes for each player idx
        globals()[f'results_idx{idx}'] = pd.DataFrame(columns=colnames)
    

    for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv')
        run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
        #run_df['title_p1'].fillna('Selfish', inplace=True)
        #run_df['title_p2'].fillna('Selfish', inplace=True)

        if 'AC' in titles: 
            run_df = run_df[~run_df['selection_player1'].isna()]

        run_df['selection_player1'] = run_df['selection_player1'].astype(int)

        run_df['selection_player1'] = run_df['selection_player1'].astype(str).str.replace('1', 'opponent 2')
        run_df['selection_player1'] = run_df['selection_player1'].str.replace('0', 'opponent 1')

        run_df['state_player1'] = run_df['state_player1'].astype(str).str.replace('1', 'D')
        run_df['state_player1'] = run_df['state_player1'].str.replace('0', 'C')

        #create for player1 & player2 for each index 
        for idx in range(population_size): 
            #globals()[f'results_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['selection_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==idx]['selection_state_player1'].astype(str)
            globals()[f'results_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['selection_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==idx]['state_player1'].astype(str)
            globals()[f'results_run_idx{idx}'] = globals()[f'results_player1_idx{idx}']

            #globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('1', 'opp2')
            #globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('0', 'opp1')
            globals()[f'results_idx{idx}'][run_name] = globals()[f'results_run_idx{idx}'] #insert result for this particular run into the overall df 
    
    #save the reformatted data
    if not os.path.isdir(f'{destination_folder}/player_selection_types'):
            os.makedirs(f'{destination_folder}/player_selection_types')
    
    for idx in range(population_size): #save results
        idx_title = run_df.loc[run_df['idx_p1']==idx]['title_p1']
        if str(idx_title[0]) not in ['AlwaysCooperate', 'AlwaysDefect']:
            idx_title = idx_title[idx_title.first_valid_index()]
            globals()[f'results_idx{idx}'].to_csv(f'{destination_folder}/player_selection_types/{idx_title}_{idx}.csv')
    

def reformat_sel_move_for_population(destination_folder, n_runs, population_size, titles):
    '''create dfs for each moral player type, cregardless of whether they acted as player1 or player2'''
    colnames = ['run'+str(i+1) for i in range(n_runs)]
    for idx in range(population_size): #initialise the results dataframes for each player idx
        globals()[f'results_idx{idx}'] = pd.DataFrame(columns=colnames)
    
    titles_short = [title.replace('QL','') for title in titles] 
    title1 = titles_short[1]
    title2 = titles_short[2]

    for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv')
        run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
        #run_df['title_p1'].fillna('Selfish', inplace=True)
        #run_df['title_p2'].fillna('Selfish', inplace=True)
        
        if 'AC' in titles: 
            run_df = run_df[~run_df['selection_player1'].isna()]

        run_df['selection_player1'] = run_df['selection_player1'].astype(int)
        run_df['selected_prev_move'] = run_df['selected_prev_move'].astype(int)

        run_df['selection_player1_clean'] = run_df['selection_player1']

        for idx in range(population_size): 
            run_df.iloc[run_df['selection_player1_clean']==int(idx)]['selection_player1_clean'] = run_df[run_df['selection_player1']==int(idx)]['selection_player1'].astype(str).str.replace(f'{idx}', f'opp. {idx+1} - {titles_short[idx]}')
        #run_df['selection_player1'] = run_df['selection_player1'].astype(str).str.replace(f'1', f'opp. 2 - {title2}')
        #run_df['selection_player1'] = run_df['selection_player1'].str.replace(f'0', f'opp. 1 - {title1}')

        run_df['selected_prev_move'] = run_df['selected_prev_move'].astype(str).str.replace('1', 'D')
        run_df['selected_prev_move'] = run_df['selected_prev_move'].str.replace('0', 'C')



        #create for player1 & player2 for each index 
        for idx in range(population_size): 
            #globals()[f'results_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['selection_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==idx]['selection_state_player1'].astype(str)
            globals()[f'results_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['selection_player1_clean'].astype(str) + ' | ' + run_df[run_df['idx_p1']==idx]['selected_prev_move'].astype(str)
            globals()[f'results_run_idx{idx}'] = globals()[f'results_player1_idx{idx}']

            #globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('1', 'opp2')
            #globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('0', 'opp1')
            globals()[f'results_idx{idx}'][run_name] = globals()[f'results_run_idx{idx}'] #insert result for this particular run into the overall df 
    
    #save the reformatted data
    if not os.path.isdir(f'{destination_folder}/player_selection_types_prevmove'):
            os.makedirs(f'{destination_folder}/player_selection_types_prevmove')
    
    for idx in range(population_size): #save results
        idx_title = run_df.loc[run_df['idx_p1']==idx]['title_p1']
        idx_title = idx_title[idx_title.first_valid_index()]
        globals()[f'results_idx{idx}'].to_csv(f'{destination_folder}/player_selection_types_prevmove/{idx_title}_{idx}.csv')
    


def plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run. NOTE we plot after iteration 0, 
    as then the agent is reacting to a default initial state, not a move from the opponent '''

    if not os.path.isdir(str(destination_folder)+'/plots'):
        os.makedirs(str(destination_folder)+'/plots')

    if not os.path.isdir(str(destination_folder)+'/plots/outcome_plots'):
        os.makedirs(str(destination_folder)+'/plots/outcome_plots')
        
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')

    ##############################################
    ### Plot from the perspective of player1 first ####

    #create action dfs for each player type: 
#    for idx in range(2):
    for idx in range(len(titles)):
        results_idx = pd.read_csv(f'{destination_folder}/player_action_types/{title_mapping[titles[idx]]}_{idx}.csv', index_col=0).sort_index()
        #results.to_csv(str(destination_folder+'/player1/action_types_full.csv'))
        results_counts = results_idx.transpose().apply(pd.value_counts).transpose()[1:] #plot after first episode, as in the first apisode they are reacting to default state=0
        results_counts.dropna(axis=1, how='all', inplace=True)
        results_counts = results_counts.reset_index().drop('index', axis=1)
        results_counts_clean = results_counts.divide(results_counts.sum(axis=1), axis=0).multiply(100)

        results_counts_clean.plot.area(figsize=(20, 6), 
                                       stacked=True, ylabel = 'Percentage agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
                                       xlabel='Iteration', linewidth=0.05,
                                       colormap='PiYG_r',
                                       #color={'C | (C,C)':'lightblue', 'C | (C,D)':'deepskyblue', 'C | (D,C)':'cyan', 'C | (D,D)':'royalblue', 'D | (C,C)':'orange', 'D | (C,D)':'yellow', 'D | (D,C)':'peru', 'D | (D,D)':'dimgrey'}, 
                                       title=f'Types of actions over time - {title_mapping[titles[idx]]} player '+str(idx)+' \n '+ 'from population: ' + str(population_list))
        plt.legend(loc=8, fontsize=20)
        plt.savefig(f'{destination_folder}/plots/action_types_area_{title_mapping[titles[idx]]}_{idx}.png', bbox_inches='tight')

def plot_selection_types_area_eachplayerinpopulation(destination_folder, titles, n_runs):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run. NOTE we plot after iteration 0, 
    as then the agent is reacting to a default initial state, not a move from the opponent '''

    if not os.path.isdir(str(destination_folder)+'/plots'):
        os.makedirs(str(destination_folder)+'/plots')

    if not os.path.isdir(str(destination_folder)+'/plots/outcome_plots'):
        os.makedirs(str(destination_folder)+'/plots/outcome_plots')
        
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')

    ##############################################
    ### Plot from the perspective of player1 first ####

    #create action dfs for each player type: 
#    for idx in range(2):
    for idx in range(len(titles)):
        results_idx = pd.read_csv(f'{destination_folder}/player_selection_types/{title_mapping[titles[idx]]}_{idx}.csv', index_col=0).sort_index()
        #results.to_csv(str(destination_folder+'/player1/action_types_full.csv'))
        results_counts = results_idx.transpose().apply(pd.value_counts).transpose()[1:] #plot after first episode, as in the first apisode they are reacting to default state=0
        results_counts.dropna(axis=1, how='all', inplace=True)
        results_counts = results_counts.reset_index().drop('index', axis=1)
        results_counts_clean = results_counts.divide(results_counts.sum(axis=1), axis=0).multiply(100)

        results_counts_clean.plot.area(figsize=(20, 6), 
                                       stacked=True, ylabel = 'Percentage agents making this type of selection \n (across '+str(n_runs)+' runs)', rot=45,
                                       xlabel='Iteration', linewidth=0.05, 
                                       colormap='PiYG_r',
                                       #color={'C | (C,C)':'lightblue', 'C | (C,D)':'deepskyblue', 'C | (D,C)':'cyan', 'C | (D,D)':'royalblue', 'D | (C,C)':'orange', 'D | (C,D)':'yellow', 'D | (D,C)':'peru', 'D | (D,D)':'dimgrey'}, 
                                       title=f'Types of selections over time - {title_mapping[titles[idx]]} player '+str(idx)+' \n '+ 'from population: ' + str(population_list))
        plt.legend(loc=8, fontsize=20)
        plt.savefig(f'{destination_folder}/plots/selection_types_area_{title_mapping[titles[idx]]}_{idx}.png', bbox_inches='tight')

def plot_selection_types_prevmove_area_eachplayerinpopulation(destination_folder, titles, n_runs):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run. NOTE we plot after iteration 0, 
    as then the agent is reacting to a default initial state, not a move from the opponent '''

    if not os.path.isdir(str(destination_folder)+'/plots'):
        os.makedirs(str(destination_folder)+'/plots')

    if not os.path.isdir(str(destination_folder)+'/plots/outcome_plots'):
        os.makedirs(str(destination_folder)+'/plots/outcome_plots')
        
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')

    title1 = titles_short[1]
    title2 = titles_short[2]

    ##############################################
    ### Plot from the perspective of player1 first ####

    #create action dfs for each player type: 
#    for idx in range(2):
    for idx in range(len(titles)):
        results_idx = pd.read_csv(f'{destination_folder}/player_selection_types_prevmove/{title_mapping[titles[idx]]}_{idx}.csv', index_col=0).sort_index()
        #results.to_csv(str(destination_folder+'/player1/action_types_full.csv'))
        results_counts = results_idx[1:].transpose().apply(pd.value_counts).transpose()[1:] #plot after first episode, as in the first apisode they are reacting to default state=0
        results_counts.dropna(axis=1, how='all', inplace=True) #drop empty columns ??
        results_counts = results_counts.reset_index().drop('index', axis=1)
        results_counts_clean = results_counts.divide(results_counts.sum(axis=1), axis=0).multiply(100)

        results_counts_clean.plot.area(figsize=(20, 6), 
                                       stacked=True, ylabel = r'%'+ ' agents making this type of selection \n (across '+str(n_runs)+' runs)', rot=45,
                                       xlabel='Iteration', linewidth=0.05, 
                                       colormap='PuOr', #'PiYG_r', 
                                       #color={'opponent 2 | C':'lightblue', 'opponent 1 | C':'deepskyblue', 'opponent 2 | D':'orange', 'opponent 1 | D':'peru'}, 
                                       #color={f'opp. 2 - {title2} | C':'lightblue', f'opp. 1 - {title1} | C':'deepskyblue', f'opp. 2 - {title2} | D':'orange', f'opp. 1 - {title1} | D':'peru'}, 
                                       #color={'C | (C,C)':'lightblue', 'C | (C,D)':'deepskyblue', 'C | (D,C)':'cyan', 'C | (D,D)':'royalblue', 'D | (C,C)':'orange', 'D | (C,D)':'yellow', 'D | (D,C)':'peru', 'D | (D,D)':'dimgrey'}, 
                                       title=f'Types of selections over time - {title_mapping[titles[idx]]} player '+str(idx)+' \n '+ 'from population: ' + str(population_list))
        plt.legend(
            #handles = ['opponent 1 | C', 'opponent 1 | D', 'opponent 2 | C', 'opponent 2 | D'], 
            #labels=[f'opp 1 - {title1} ~ C', f'opp. 1 - {title1}| D', f'opp. 2 - {title2}| C', f'opp. 2 - {title2}| D'], 
            loc=8, fontsize=15)
        plt.savefig(f'{destination_folder}/plots/selection_types__prevmove_area_{title_mapping[titles[idx]]}_{idx}.png', bbox_inches='tight')


if False: 
    def reformat_a_for_population_6(destination_folder, n_runs, population_size=6):
        '''create dfs for each moral player type, cregardless of whether they acted as player1 or player2'''
        colnames = ['run'+str(i+1) for i in range(n_runs)]
        results_idx0 = pd.DataFrame(columns=colnames)
        results_idx1 = pd.DataFrame(columns=colnames)
        results_idx2 = pd.DataFrame(columns=colnames)
        results_idx3 = pd.DataFrame(columns=colnames)
        results_idx4 = pd.DataFrame(columns=colnames)
        results_idx5 = pd.DataFrame(columns=colnames)

        for run in os.listdir(str(destination_folder+'/history')):
            run_name = str(run).strip('.csv')
            run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)))#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
            run_df['title_p1'].fillna('Selfish', inplace=True)
            run_df['title_p2'].fillna('Selfish', inplace=True)

            #create for player1 & player2 for each index 
            results_player1_idx0 = run_df[run_df['idx_p1']==0]['action_player1'].astype(str)
            results_player2_idx0 = run_df[run_df['idx_p2']==0]['action_player2'].astype(str)
            results_run_idx0 = results_player1_idx0.append(results_player2_idx0)
            results_run_idx0 = results_run_idx0.str.replace('1', 'D')
            results_run_idx0 = results_run_idx0.str.replace('0', 'C')
            results_idx0[run_name] = results_run_idx0 #insert result for this particular run into the overall df 

            results_player1_idx1 = run_df[run_df['idx_p1']==1]['action_player1'].astype(str)
            results_player2_idx1 = run_df[run_df['idx_p2']==1]['action_player2'].astype(str)
            results_run_idx1 = results_player1_idx1.append(results_player2_idx1)
            results_run_idx1 = results_run_idx1.str.replace('1', 'D')
            results_run_idx1 = results_run_idx1.str.replace('0', 'C')
            results_idx1[run_name] = results_run_idx1

            results_player1_idx2 = run_df[run_df['idx_p1']==2]['action_player1'].astype(str)
            results_player2_idx2 = run_df[run_df['idx_p2']==2]['action_player2'].astype(str)
            results_run_idx2 = results_player1_idx2.append(results_player2_idx2)
            results_run_idx2 = results_run_idx2.str.replace('1', 'D')
            results_run_idx2 = results_run_idx2.str.replace('0', 'C')
            results_idx2[run_name] = results_run_idx2

            results_player1_idx3 = run_df[run_df['idx_p1']==3]['action_player1'].astype(str)
            results_player2_idx3 = run_df[run_df['idx_p2']==3]['action_player2'].astype(str)
            results_run_idx3 = results_player1_idx3.append(results_player2_idx3)
            results_run_idx3 = results_run_idx3.str.replace('1', 'D')
            results_run_idx3 = results_run_idx3.str.replace('0', 'C')
            results_idx3[run_name] = results_run_idx3

            results_player1_idx4 = run_df[run_df['idx_p1']==4]['action_player1'].astype(str)
            results_player2_idx4 = run_df[run_df['idx_p2']==4]['action_player2'].astype(str)
            results_run_idx4 = results_player1_idx4.append(results_player2_idx4)
            results_run_idx4 = results_run_idx4.str.replace('1', 'D')
            results_run_idx4 = results_run_idx4.str.replace('0', 'C')
            results_idx4[run_name] = results_run_idx4

            results_player1_idx5 = run_df[run_df['idx_p1']==5]['action_player1'].astype(str)
            results_player2_idx5 = run_df[run_df['idx_p2']==5]['action_player2'].astype(str)
            results_run_idx5 = results_player1_idx5.append(results_player2_idx5)
            results_run_idx5 = results_run_idx5.str.replace('1', 'D')
            results_run_idx5 = results_run_idx5.str.replace('0', 'C')
            results_idx5[run_name] = results_run_idx5
        
        #save the reformatted data
        if not os.path.isdir(f'{destination_folder}/player_actions'):
                os.makedirs(f'{destination_folder}/player_actions')
        
        idx0_title = run_df.loc[run_df['idx_p1']==0]['title_p1']
        idx0_title = idx0_title[idx0_title.first_valid_index()]
        idx1_title = run_df.loc[run_df['idx_p1']==1]['title_p1']
        idx1_title = idx1_title[idx1_title.first_valid_index()]
        idx2_title = run_df.loc[run_df['idx_p1']==2]['title_p1']
        idx2_title = idx2_title[idx2_title.first_valid_index()]
        idx3_title = run_df.loc[run_df['idx_p1']==3]['title_p1']
        idx3_title = idx3_title[idx3_title.first_valid_index()]
        idx4_title = run_df.loc[run_df['idx_p1']==4]['title_p1']
        idx4_title = idx4_title[idx4_title.first_valid_index()]
        idx5_title = run_df.loc[run_df['idx_p1']==5]['title_p1']
        idx5_title = idx5_title[idx5_title.first_valid_index()]

        results_idx0.to_csv(f'{destination_folder}/player_actions/{idx0_title}_0.csv')
        results_idx1.to_csv(f'{destination_folder}/player_actions/{idx1_title}_1.csv')
        results_idx2.to_csv(f'{destination_folder}/player_actions/{idx2_title}_2.csv')
        results_idx3.to_csv(f'{destination_folder}/player_actions/{idx3_title}_3.csv')
        results_idx4.to_csv(f'{destination_folder}/player_actions/{idx4_title}_4.csv')
        results_idx5.to_csv(f'{destination_folder}/player_actions/{idx5_title}_5.csv')


    def reformat_a_s_for_population_6(destination_folder, n_runs, population_size=6):
        '''create dfs for each moral player type, cregardless of whether they acted as player1 or player2'''
        colnames = ['run'+str(i+1) for i in range(n_runs)]
        results_idx0 = pd.DataFrame(columns=colnames)
        results_idx1 = pd.DataFrame(columns=colnames)
        results_idx2 = pd.DataFrame(columns=colnames)
        results_idx3 = pd.DataFrame(columns=colnames)
        results_idx4 = pd.DataFrame(columns=colnames)
        results_idx5 = pd.DataFrame(columns=colnames)

        for run in os.listdir(str(destination_folder+'/history')):
            run_name = str(run).strip('.csv')
            run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)))#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
            run_df['title_p1'].fillna('Selfish', inplace=True)
            run_df['title_p2'].fillna('Selfish', inplace=True)

            #create for player1 & player2 for each index 
            results_player1_idx0 = run_df[run_df['idx_p1']==0]['action_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==0]['state_player1'].astype(str)
            results_player2_idx0 = run_df[run_df['idx_p2']==0]['action_player2'].astype(str) + ' | ' + run_df[run_df['idx_p2']==0]['state_player2'].astype(str)
            results_run_idx0 = results_player1_idx0.append(results_player2_idx0)
            results_run_idx0 = results_run_idx0.str.replace('1', 'D')
            results_run_idx0 = results_run_idx0.str.replace('0', 'C')
            results_idx0[run_name] = results_run_idx0 #insert result for this particular run into the overall df 

            results_player1_idx1 = run_df[run_df['idx_p1']==1]['action_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==1]['state_player1'].astype(str)
            results_player2_idx1 = run_df[run_df['idx_p2']==1]['action_player2'].astype(str) + ' | ' + run_df[run_df['idx_p2']==1]['state_player2'].astype(str)
            results_run_idx1 = results_player1_idx1.append(results_player2_idx1)
            results_run_idx1 = results_run_idx1.str.replace('1', 'D')
            results_run_idx1 = results_run_idx1.str.replace('0', 'C')
            results_idx1[run_name] = results_run_idx1

            results_player1_idx2 = run_df[run_df['idx_p1']==2]['action_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==2]['state_player1'].astype(str)
            results_player2_idx2 = run_df[run_df['idx_p2']==2]['action_player2'].astype(str) + ' | ' + run_df[run_df['idx_p2']==2]['state_player2'].astype(str)
            results_run_idx2 = results_player1_idx2.append(results_player2_idx2)
            results_run_idx2 = results_run_idx2.str.replace('1', 'D')
            results_run_idx2 = results_run_idx2.str.replace('0', 'C')
            results_idx2[run_name] = results_run_idx2

            results_player1_idx3 = run_df[run_df['idx_p1']==3]['action_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==3]['state_player1'].astype(str)
            results_player2_idx3 = run_df[run_df['idx_p2']==3]['action_player2'].astype(str) + ' | ' + run_df[run_df['idx_p2']==3]['state_player2'].astype(str)
            results_run_idx3 = results_player1_idx3.append(results_player2_idx3)
            results_run_idx3 = results_run_idx3.str.replace('1', 'D')
            results_run_idx3 = results_run_idx3.str.replace('0', 'C')
            results_idx3[run_name] = results_run_idx3

            results_player1_idx4 = run_df[run_df['idx_p1']==4]['action_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==4]['state_player1'].astype(str)
            results_player2_idx4 = run_df[run_df['idx_p2']==4]['action_player2'].astype(str) + ' | ' + run_df[run_df['idx_p2']==4]['state_player2'].astype(str)
            results_run_idx4 = results_player1_idx4.append(results_player2_idx4)
            results_run_idx4 = results_run_idx4.str.replace('1', 'D')
            results_run_idx4 = results_run_idx4.str.replace('0', 'C')
            results_idx4[run_name] = results_run_idx4

            results_player1_idx5 = run_df[run_df['idx_p1']==5]['action_player1'].astype(str) + ' | ' + run_df[run_df['idx_p1']==5]['state_player1'].astype(str)
            results_player2_idx5 = run_df[run_df['idx_p2']==5]['action_player2'].astype(str) + ' | ' + run_df[run_df['idx_p2']==5]['state_player2'].astype(str)
            results_run_idx5 = results_player1_idx5.append(results_player2_idx5)
            results_run_idx5 = results_run_idx5.str.replace('1', 'D')
            results_run_idx5 = results_run_idx5.str.replace('0', 'C')
            results_idx5[run_name] = results_run_idx5
        
        #save the reformatted data
        if not os.path.isdir(f'{destination_folder}/player_action_types'):
                os.makedirs(f'{destination_folder}/player_action_types')
        
        idx0_title = run_df.loc[run_df['idx_p1']==0]['title_p1']
        idx0_title = idx0_title[idx0_title.first_valid_index()]
        idx1_title = run_df.loc[run_df['idx_p1']==1]['title_p1']
        idx1_title = idx1_title[idx1_title.first_valid_index()]
        idx2_title = run_df.loc[run_df['idx_p1']==2]['title_p1']
        idx2_title = idx2_title[idx2_title.first_valid_index()]
        idx3_title = run_df.loc[run_df['idx_p1']==3]['title_p1']
        idx3_title = idx3_title[idx3_title.first_valid_index()]
        idx4_title = run_df.loc[run_df['idx_p1']==4]['title_p1']
        idx4_title = idx4_title[idx4_title.first_valid_index()]
        idx5_title = run_df.loc[run_df['idx_p1']==5]['title_p1']
        idx5_title = idx5_title[idx5_title.first_valid_index()]

        results_idx0.to_csv(f'{destination_folder}/player_action_types/{idx0_title}_0.csv')
        results_idx1.to_csv(f'{destination_folder}/player_action_types/{idx1_title}_1.csv')
        results_idx2.to_csv(f'{destination_folder}/player_action_types/{idx2_title}_2.csv')
        results_idx3.to_csv(f'{destination_folder}/player_action_types/{idx3_title}_3.csv')
        results_idx4.to_csv(f'{destination_folder}/player_action_types/{idx4_title}_4.csv')
        results_idx5.to_csv(f'{destination_folder}/player_action_types/{idx5_title}_5.csv')
        

def reformat_a_for_population(destination_folder, n_runs, population_size, num_iter, long_titles, episodes=True):
        '''create dfs for each moral player type, regardless of whether they acted as player1 or player2'''
        colnames = ['run'+str(i+1) for i in range(n_runs)]

        if episodes:
            num_iter_new = num_iter * population_size
        else: 
            num_iter_new = num_iter

        for idx in range(population_size): #initialise the results dataframes for each player idx
            globals()[f'results_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))
        
        path = str(destination_folder+'/history/'+'*.csv')
        
        for run in glob.glob(path): #os.listdir():
            print(f'proecssing run {run}')
            run_name = str(run).replace(str(destination_folder)+'/history/', '')
            if '.csv' in run_name:
                run_name = str(run_name).strip('.csv')
            #try: 
                run_df = pd.read_csv(run)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
                #run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)))#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
                
                #run_df['title_p1'].fillna('Selfish', inplace=True)
                #run_df['title_p2'].fillna('Selfish', inplace=True)
                run_df = run_df[run_df['idx_p1']!=run_df['idx_p2']] #drop self-selections - FOR EARLIER VERSIONS
            #except: 
            #    break 

                #create for player1 & player2 for each index 
                for idx in range(population_size): 
                    globals()[f'results_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['action_player1'].astype(str)
                    globals()[f'results_player2_idx{idx}'] = run_df[run_df['idx_p2']==idx]['action_player2'].astype(str)
                    globals()[f'results_run_idx{idx}'] = globals()[f'results_player1_idx{idx}'].append(globals()[f'results_player2_idx{idx}'])

                    globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('1', 'D')
                    globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('0', 'C')
                    globals()[f'results_idx{idx}'][run_name] = globals()[f'results_run_idx{idx}'] #insert result for this particular run into the overall df 
            
        #save the reformatted data
        if not os.path.isdir(f'{destination_folder}/player_actions'):
                os.makedirs(f'{destination_folder}/player_actions')

        print('saving action csvs for each player_idx to disc')
        for idx in range(population_size): #save results
            #idx_title = run_df.loc[run_df['idx_p1']==idx]['title_p1']
            #idx_title = idx_title[idx_title.first_valid_index()]
            idx_title = long_titles[idx]
            globals()[f'results_idx{idx}'].to_csv(f'{destination_folder}/player_actions/{idx_title}_{idx}.csv')

if False:    
    def reformat_sel_for_population(destination_folder, n_runs, population_size):
            '''create dfs for each moral player type's selections'''
            colnames = ['run'+str(i+1) for i in range(n_runs)]

            for idx in range(population_size): #initialise the results dataframes for each player idx
                globals()[f'results_idx{idx}'] = pd.DataFrame(columns=colnames)
            
            path = str(destination_folder+'/history/'+'*.csv')
            
            for run in glob.glob(path): #os.listdir():
                print(f'proecssing run {run}')
                run_name = str(run).replace(str(destination_folder)+'/history/', '')
                run_name = str(run_name).strip('.csv')
                run_df = pd.read_csv(run)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
                #run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)))#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
                #run_df['title_p1'].fillna('Selfish', inplace=True)
                #run_df['title_p2'].fillna('Selfish', inplace=True)
                #run_df = run_df[run_df['idx_p1']!=run_df['idx_p2']] #drop self-selections - FOR EARLIER VERSIONS

                #create for player1 & player2 for each index 
                for idx in range(population_size): 
                    globals()[f'results_run_idx{idx}'] = run_df[run_df['idx_p1']==idx]['selection_player1'].astype(str)
                    #globals()[f'results_player2_idx{idx}'] = run_df[run_df['idx_p2']==idx]['selection_player2'].astype(str)
                    #globals()[f'results_run_idx{idx}'] = globals()[f'results_player1_idx{idx}'].append(globals()[f'results_player2_idx{idx}'])

                    #globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('1', 'D')
                    #globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('0', 'C')
                    globals()[f'results_idx{idx}'][run_name] = globals()[f'results_run_idx{idx}'] #insert result for this particular run into the overall df 
            
            #save the reformatted data
            if not os.path.isdir(f'{destination_folder}/player_selections'):
                    os.makedirs(f'{destination_folder}/player_selections')

            for idx in range(population_size): #save results
                idx_title = run_df.loc[run_df['idx_p1']==idx]['title_p1']
                idx_title = idx_title[idx_title.first_valid_index()]
                globals()[f'results_idx{idx}'].to_csv(f'{destination_folder}/player_selections/{idx_title}_{idx}.csv')



def plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, plot_defection=False):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating? '''
    
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')

    if population_size == 2: 
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(6*population_size,2.5))
        linewidth = 0.06
    elif population_size == 3: 
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(6*population_size,2.5))
        linewidth = 0.05
    elif population_size == 4: 
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, sharex=True, figsize=(6*population_size,2.5))
        linewidth = 0.03
    elif population_size == 5: 
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(6*population_size,2.5))
        linewidth = 0.01
    elif population_size == 7: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(nrows=1, ncols=7, sharey=True, sharex=True, figsize=(3*population_size,2.5))
        linewidth = 0.01
    elif population_size == 10: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(3*population_size,2.5))
        linewidth = 0.005
    elif population_size == 20: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(3*population_size,2.5))
        linewidth = 0.005 #0.0005

    #plot for each player type: 
    for idx in range(len(titles)):
        title = title_mapping[titles[idx]]
        if title in ['AlwaysCooperate', 'AlwaysDefect', 'GilbertElliot_typeb']:
            linewidth_new = 1.5
        else: 
            linewidth_new = linewidth

        actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
        results_idx = pd.DataFrame(index=actions_idx.index)

        #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
        results_idx['%_cooperate'] = actions_idx[actions_idx[:]=='C'].count(axis='columns') / actions_idx.count(axis='columns') * 100


        ax = locals()["ax"+str(idx+1)]
        ax.set_title(f'\n {title} \n player{idx}') #f'\n {title} player{idx}'
        #plt.title(f'Cooperation by {title} player{idx} \n from population {population_list}') #(\n percent cooperated over '+str(n_runs)+r' runs)'

        #plot results 
        ax.plot(results_idx.index[:], results_idx['%_cooperate'], color='darkgreen', linewidth=linewidth_new) #label=f'{title}_{idx}', 
        ax.set_ylim([-10, 110])
        ax.set(xlabel='Iteration', ylabel='') #ylabel='Percentage cooperating \n (over '+str(n_runs)+r' runs)'

        #plt.savefig(f'{destination_folder}/plots/cooperation_{title}_{idx}.png', bbox_inches='tight')


        if False: #plotting separate plots for each player 
            #plot results 
            plt.figure(dpi=60) #figsize=(10, 6), 
            plt.plot(results_idx.index[:], results_idx['%_cooperate'], label=f'{title}_{idx}', color='darkgreen', linewidth=0.06)
            #plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
            #plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
            #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            plt.title(f'Cooperation by {title} player{idx} \n from population {population_list}') #(\n percent cooperated over '+str(n_runs)+r' runs)'
            plt.gca().set_ylim([-10, 110])
            plt.ylabel('Percentage cooperating \n (over '+str(n_runs)+r' runs)')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/cooperation_{title}_{idx}.png', bbox_inches='tight')

        if plot_defection:
            results_idx['%_defect'] = actions_idx[actions_idx[:]=='D'].count(axis='columns') / actions_idx.count(axis='columns') * 100

            #plot results 
            plt.figure(dpi=60) #figsize=(10, 6), 
            plt.plot(results_idx.index[:], results_idx['%_defect'], label=f'{title}_{idx}', color='red', linewidth=linewidth)
            #plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
            #plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
            #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            plt.title(f'Defection by {title} player{idx} \n from population {population_list}') #(\n percent cooperated over '+str(n_runs)+r' runs)'
            plt.gca().set_ylim([-10, 110])
            plt.ylabel('Percentage defecting \n (over '+str(n_runs)+r' runs)')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/defection_{title}_{idx}.png', bbox_inches='tight')

    ax1.set(ylabel=r'% cooperating'+ f'\n(over {n_runs} runs)')
    #plt.ylabel(r'% cooperating'+ f'\n(over {n_runs} runs)')
    plt.suptitle(f'% cooperating for each player from population {population_list} \n', fontsize=28, y=1.4)

    plt.savefig(f'{destination_folder}/plots/cooperation_eachplayer.png', bbox_inches='tight')

def plot_actions_eachplayertype(destination_folder, titles, n_runs, plot_defection):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating? '''
    
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')

    types = set(titles)

    #plot for each player type: 
    #plot for each player type: 
    for type in set(titles): 
        results_type = pd.DataFrame()

        type_long = title_mapping[type]

        path = str(destination_folder+'/player_actions/'+type_long+'*.csv')

        for file in glob.glob(path): 
            print(f'proecssing type {type_long}, file {file}')
            #run_name = str(run).strip('.csv')
            actions_idx = pd.read_csv(file, index_col=0).sort_index()
            print(actions_idx.head())





    for idx in range(len(titles)):
        title = title_mapping[titles[idx]]

        actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
        results_idx = pd.DataFrame(index=actions_idx.index)

        #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
        results_idx['%_cooperate'] = actions_idx[actions_idx[:]=='C'].count(axis='columns') / actions_idx.count(axis='columns') * 100

        #convert to %
        #results_idx['%_cooperate'] = (results_idx['%_cooperate']/results_idx.count(axis='columns'))*100
        #results_idx['%_defect'] = (results_idx['%_defect']/results_idx.count(axis='columns'))*100

        #plot results 
        plt.figure(dpi=60) #figsize=(10, 6), 
        plt.plot(results_idx.index[:], results_idx['%_cooperate'], label=f'{title}_{idx}', color='blue', linewidth=0.06)
        #plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
        #plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
        #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
        plt.title(f'Cooperation by {title} player{idx} \n from population {population_list}') #(\n percent cooperated over '+str(n_runs)+r' runs)'
        plt.gca().set_ylim([0, 100])
        plt.ylabel('Percentage cooperating \n (over '+str(n_runs)+r' runs)')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'{destination_folder}/plots/cooperation_{title}_{idx}.png', bbox_inches='tight')

        if plot_defection:
            results_idx['%_defect'] = actions_idx[actions_idx[:]=='D'].count(axis='columns') / actions_idx.count(axis='columns') * 100

            #plot results 
            plt.figure(dpi=60) #figsize=(10, 6), 
            plt.plot(results_idx.index[:], results_idx['%_defect'], label=f'{title}_{idx}', color='red', linewidth=0.06)
            #plt.plot(my_df_player2.index[:], means_player2[:], label=f'player2 - {player2_title}', color='orange')
            #plt.fill_between(my_df_player1.index[:], means_player1-sds_player1, means_player1+sds_player1, facecolor='#95d0fc', alpha=0.7)
            #plt.fill_between(my_df_player2.index[:], means_player2-sds_player2, means_player2+sds_player2, facecolor='#fed8b1', alpha=0.7)
            plt.title(f'Defection by {title} player{idx} \n from population {population_list}') #(\n percent cooperated over '+str(n_runs)+r' runs)'
            plt.gca().set_ylim([0, 100])
            plt.ylabel('Percentage defecting \n (over '+str(n_runs)+r' runs)')
            plt.xlabel('Iteration')
            leg = plt.legend() # get the legend object
            for line in leg.get_lines(): # change the line width for the legend
                line.set_linewidth(4.0)
            plt.savefig(f'{destination_folder}/plots/defection_{title}_{idx}.png', bbox_inches='tight')

def plot_cooperation_population(destination_folder, titles, n_runs, num_iter, with_CI=True):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating? '''
    
    if len(titles)<=6:
        titles_short = [title.replace('QL','') for title in titles] 
        temp = ''
        for t in titles_short: 
            temp += (t+',')
        population_list = temp.strip(',')
    else:  
        population_list = destination_folder.replace('results/', '')

    #combine each player's actions into one df: 
    result_population = pd.DataFrame(index=range(num_iter)) 

    for idx in range(len(titles)):
        title = title_mapping[titles[idx]]
        actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
        results_idx = pd.DataFrame(index=actions_idx.index)

        #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
        results_idx['%_cooperate'] = actions_idx[actions_idx[:]=='C'].count(axis='columns') / actions_idx.count(axis='columns') * 100
        #results_idx['%_defect'] = actions_idx[actions_idx[:]=='D'].count(axis='columns') / actions_idx.count(axis='columns') * 100

        result_population[f'{title}_{idx}'] = results_idx['%_cooperate']
    
    if with_CI:
        means = result_population.mean(axis=1)
        sds = result_population.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        result_population['%_cooperate_mean'] = means
        result_population['%_cooperate_ci'] = ci
    else: 
        result_population['%_cooperate_mean'] = result_population.mean(axis=1)


    #plot results 
    plt.figure(dpi=80, figsize=(25, 6)) 
    if with_CI:
                plt.plot(result_population.index[:], result_population['%_cooperate_mean'], label=f'{population_list}', color='darkgreen', linewidth=0.04)
                plt.fill_between(result_population.index[:], result_population['%_cooperate_mean']-result_population['%_cooperate_ci'], result_population['%_cooperate_mean']+result_population['%_cooperate_ci'], facecolor='lightgreen', linewidth=0.04, alpha=1)
    else: 
        plt.plot(result_population.index[:], result_population['%_cooperate_mean'], label=f'{population_list}', color='darkgreen', linewidth=0.05)

    plt.title(f'Cooperation in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([0, 100])
    plt.ylabel(r'% cooperating'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)' + "\n (mean across all players' percentage C)")
    plt.xlabel('Iteration')
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)

    #save the reformatted data
    if not os.path.isdir(f'{destination_folder}/plots'):
            os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/cooperation_{population_list}.png', bbox_inches='tight')

def plot_cooperation_population_v2(destination_folder, titles, n_runs, num_iter, with_CI=True, reduced=False):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating? '''
    
    if len(titles)<=6:
        titles_short = [title.replace('QL','') for title in titles] 
        temp = ''
        for t in titles_short: 
            temp += (t+',')
        population_list = temp.strip(',')
    else:  
        population_list = destination_folder.replace('results/', '')

    #combine each player's actions into one df: 
    result_population = pd.DataFrame(index=range(num_iter)) 

    for run_idx in range(n_runs):
        run_idx += 1
        run_df = pd.read_csv(destination_folder+f'/history/run{run_idx}.csv', index_col=0)[['action_player1', 'action_player2']]
        run_df[f'%_c_run{run_idx}'] = run_df[run_df[:]==0].count(axis='columns') / run_df.count(axis='columns') * 100
        result_population[f'run{run_idx}'] = run_df[f'%_c_run{run_idx}']


    if with_CI:
        means = result_population.mean(axis=1)
        sds = result_population.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        result_population['%_cooperate_mean'] = means
        result_population['%_cooperate_ci'] = ci
    else: 
        result_population['%_cooperate_mean'] = result_population.mean(axis=1)

    if reduced: 
        #keep every 5th row 
        result_population = result_population[result_population.index % 5 == 0]
        reduced_option = ' (every 5)'
    else: 
        reduced_option = None

    #plot results 
    plt.figure(dpi=80, figsize=(25, 6)) 
    if with_CI:
                plt.plot(result_population.index[:], result_population['%_cooperate_mean'], label=f'{population_list}', color='darkgreen', linewidth=0.05)
                plt.fill_between(result_population.index[:], result_population['%_cooperate_mean']-result_population['%_cooperate_ci'], result_population['%_cooperate_mean']+result_population['%_cooperate_ci'], facecolor='lightgreen', linewidth=0.04, alpha=1)
    else: 
        plt.plot(result_population.index[:], result_population['%_cooperate_mean'], label=f'{population_list}', color='darkgreen', linewidth=0.05)

    if reduced_option: 
        plt.title(f'Cooperation in population \n {population_list} '+ reduced_option) 
    else: 
        plt.title(f'Cooperation in population \n {population_list} ') 
    plt.gca().set_ylim([0, 100])
    plt.ylabel(r'% cooperating'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)' + "\n (mean across all players' percentage C)")
    plt.xlabel('Iteration')


    if not os.path.isdir(f'{destination_folder}/plots'):
        os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/cooperation_{population_list}_V2.png', bbox_inches='tight')


def plot_cooperative_selections(destination_folder, titles, n_runs, num_iter, iter_range=None):
    '''create cooperative_selection bool variable for each run, then plot % cooperative selections'''
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')
    
    colnames = ['run'+str(i+1) for i in range(n_runs)]

    data = pd.DataFrame(columns=colnames)
    if iter_range: 
        results_idx = pd.DataFrame(index=list(range(iter_range[0], iter_range[1])))
    else: 
        results_idx = pd.DataFrame(index=list(range(num_iter)))
    
    for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv')
        run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
        #drop NAs from the df - i.e. drop rows where a selection was not actually made 
        run_df = run_df[run_df['selected_prev_move'].notna()]
        if iter_range: 
            run_df = run_df.iloc[iter_range[0]:iter_range[1]]
        run_df['cooperative_selection'] = run_df['selected_prev_move']==0 #.apply(lambda x: int(x))
        data[run_name] = run_df['cooperative_selection']

    results_idx['cooperative_selection'] = data[data[:]==True].count(axis='columns') / data.count(axis='columns') * 100


    #plot results 
    plt.figure(dpi=80, figsize=(25, 6))
    plt.plot(results_idx.dropna().index[:], results_idx.dropna()['cooperative_selection'], label='cooperative selections', color='blue', linewidth=0.1)
    plt.title(f'Cooperative selections in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-5, 105])
    plt.ylabel("% cooperative partners selected \n (those that C in last move) \n (over "+str(n_runs)+r" runs)")
    plt.xlabel('Iteration (where Selfish player made a selection)')
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)

    #save the reformatted data
    if not os.path.isdir(f'{destination_folder}/plots'):
            os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/cooperative_selections_{population_list}.png', bbox_inches='tight')

def plot_cooperative_selections_whereavailable(destination_folder, titles, n_runs):
    '''create cooperative_selection bool variable for each run, then plot % cooperative selections'''
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')
    
    runnames_imtermediate = ['run'+str(i+1) for i in range(n_runs)]
    data_selections = pd.DataFrame(columns=runnames_imtermediate)
    data_availables = pd.DataFrame(columns=runnames_imtermediate)

    path = str(destination_folder+'/history/'+'*.csv')
        
    for run in glob.glob(path): #for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv').split('/')[-1]
        try: 
            #run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
            run_df = pd.read_csv(str(run), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
            #drop NAs from the df - i.e. drop rows where a selection was not actually made 
            run_df_clean = run_df[run_df['selected_prev_move'].notna()]

            data_selections[run_name] = run_df_clean['selected_prev_move']==0
            data_availables[run_name] = run_df_clean['cooperative_sel_available']==True
        except:
            print('failed at run :', run_name)
            break 


    data_selections['sum_across_runs'] = data_selections.sum(axis=1)
    data_availables['sum_across_runs'] = data_availables.sum(axis=1)

    result = data_selections['sum_across_runs']/data_availables['sum_across_runs'] * 100

    if len(titles) < 3:
        if 'AC' in titles: 
            linewidth=0.9
        elif 'AD' in titles:
            linewidth=0.9
        else: 
            linewidth = 0.05
    else: 
        linewidth = 0.05

    #plot results 
    plt.figure(dpi=80, figsize=(25, 6))
    #plt.plot(result.index[:], result[:], label=r'% cooperative selections', color='blue', linewidth=linewidth)
    plt.plot(result, label=r'% cooperative selections', color='blue', linewidth=linewidth)
    plt.title(f'Cooperative selections in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-5, 105])
    plt.ylabel("% cooperative partners selected \n (those that C in last move) \n over all cooperative partners available  \n (over "+str(n_runs)+r" runs)")
    plt.xlabel('Iteration (where a learning player made a selection)')
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)

    #save the reformatted data
    if not os.path.isdir(f'{destination_folder}/plots'):
            os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/cooperative_selections_{population_list}_whereavailable.png', bbox_inches='tight')


    #plot cooperative selections available - absolute number out of n_runs
    plt.figure(dpi=80, figsize=(25, 6))
    #plt.plot(data_availables['sum_across_runs'].index[:], data_availables['sum_across_runs'][:], label=r'num cooperative opponents available', color='indigo', linewidth=0.07)
    plt.plot(data_availables['sum_across_runs'], label=r'num cooperative opponents available', color='indigo', linewidth=0.07)
    plt.title(f'Cooperative partners available in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-0.5, n_runs+0.5])
    plt.ylabel("num cooperative partners available \n (those that C in last move)  \n (over "+str(n_runs)+r" runs)")
    plt.xlabel('Iteration (where Selfish player made a selection)')

    #plot cooperative selections made - absolute number out of n_runs
    plt.figure(dpi=80, figsize=(25, 6))
    #plt.plot(data_selections['sum_across_runs'].index[:], data_selections['sum_across_runs'][:], label=r'num cooperative selections made', color='blueviolet', linewidth=0.07)
    plt.plot(data_selections['sum_across_runs'], label=r'num cooperative selections made', color='blueviolet', linewidth=0.07)
    plt.title(f'Cooperative selections made in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-0.5, n_runs+0.5])
    plt.ylabel("num cooperative selections made  \n (those that C in last move)  \n (over "+str(n_runs)+r" runs)")
    plt.xlabel('Iteration (where Selfish player made a selection)')



#plots for player1 & players2 (i.e. mixing different player types)
def plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced, option=None, combine_CDDC=False):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run.
    NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent'''
    if not os.path.isdir(str(destination_folder)+'/plots'):
        os.makedirs(str(destination_folder)+'/plots')

    if not os.path.isdir(str(destination_folder)+'/plots/outcome_plots'):
        os.makedirs(str(destination_folder)+'/plots/outcome_plots')

    colnames = ['run'+str(i+1) for i in range(n_runs)]
    actions_player1 = pd.DataFrame(columns=colnames)
    actions_player2 = pd.DataFrame(columns=colnames)

    for run in os.listdir(str(destination_folder+'/history')):
        if '.csv' in run:
            run_name = str(run).strip('.csv')
            run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
            actions_player1[run_name] = run_df['action_player1']
            actions_player2[run_name] = run_df['action_player2']

    results = pd.DataFrame(columns=colnames)

    for colname in colnames: #loop over every run
        str_value = actions_player1[colname].astype(str) + ', ' + actions_player2[colname].astype(str)
        str_value = str_value.str.replace('1', 'D')
        str_value = str_value.str.replace('0', 'C')
        str_value = str_value.str.replace(' ', '')
        if combine_CDDC:
            str_value = str_value.replace('C,D', 'C,D/D,C').replace('D,C', 'C,D/D,C') #when the order of actions does not matter, combine CD/DC into one color
        results[colname] = str_value

    results.to_csv(str(destination_folder+'/action_pairs.csv'))

    if reduced: 
        #keep every 10th row 
        results = results[results.index % 10 == 0]
        reduced_option = ' (every 10)'
    else: 
        reduced_option = None

    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first apisode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    titles = [title.replace('QL','') for title in titles] 
    titles = [title.replace('Ethics','').replace('_','-') for title in titles] 
    temp = ''
    for t in titles: 
        temp += (t+',')
    temp = temp.strip(',')
    if len(titles)>6: 
        temp = destination_folder.replace('results/', '')

    #plt.figure(figsize=(20, 15), dpi=100)
    if num_iter > 20000:
        plt.figure(dpi=80, figsize=(10, 4))
    else:
        plt.figure(dpi=80, figsize=(5, 4))
    
    if combine_CDDC:
        color = {'C,C':'#28641E', 'C,D/D,C':'#FFFFFF', 'D,D':'#8E0B52'}
    else: 
        color = {'C,C':'#28641E', 'C,D':'#B0DC82', 'D,C':'#EEAED4', 'D,D':'#8E0B52'}

    plt.rcParams.update({'font.size':20})
    results_counts.plot.area(stacked=True, ylabel = '# action pairs obs.', #rot=45,
        xlabel='Iteration (2 players at any one time)', #colormap='PiYG_r',
        color=color, linewidth=0.05, alpha=0.9,
        #color={'C, C':'#28641E', 'C, D':'#B0DC82', 'D, C':'#EEAED4', 'D, D':'#8E0B52'}, linewidth=0.05, alpha=0.9,
        #color={'C, C':'royalblue', 'C, D':'lightblue', 'D, C':'yellow', 'D, D':'orange'},
        title = str('Population: ' + temp + reduced_option)
        ).legend(fontsize='x-small', loc='center') #Pairs of simultaneous actions over time: \n '+

    #plt.savefig(f'{destination_folder}/plots/action_pairs.png', bbox_inches='tight')
    if not os.path.isdir('results/outcome_plots/actions'):
            os.makedirs('results/outcome_plots/actions')

    if not option: #if plotting main results
        population = destination_folder.split('/')[1]

    else: #if plotting extra parameter search for beta in QLVM 
        population = destination_folder
        #pair += str(option)
    
    plt.savefig(f'results/outcome_plots/actions/popul_{population}.pdf', bbox_inches='tight')


def plot_action_pairs_population(destination_folder, titles, n_runs, option=None):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run.
    NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent'''
    if not os.path.isdir(str(destination_folder)+'/plots'):
        os.makedirs(str(destination_folder)+'/plots')

    if not os.path.isdir(str(destination_folder)+'/plots/outcome_plots'):
        os.makedirs(str(destination_folder)+'/plots/outcome_plots')

    if False: #if os.path.exists(str(destination_folder+'/action_pairs.csv')):
        results = pd.read_csv(str(destination_folder+'/action_pairs.csv'), index_col=0)

    else: #if action_pairs.csv does not yet exist 
        actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)
        actions_player2 = pd.read_csv(f'{destination_folder}/player2/action.csv', index_col=0)

        #rename columns 
        colnames = ['run'+str(i) for i in range(n_runs)]
        actions_player1.columns = colnames
        actions_player2.columns = colnames

        results = pd.DataFrame(columns=colnames)

        for colname in colnames: #loop over every run
            str_value = actions_player1[colname].astype(str) + ', ' + actions_player2[colname].astype(str)
            str_value = str_value.str.replace('1', 'D')
            str_value = str_value.str.replace('0', 'C')
            str_value = str_value.replace('C, D', 'C,D/D,C').replace('D, C', 'C,D/D,C') #the order of actions does not matter, so comibe CD/DC into one 
            results[colname] = str_value

        results.to_csv(str(destination_folder+'/action_pairs.csv'))

    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first apisode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    titles = [title.replace('QL','') for title in titles] 
    titles = [title.replace('Ethics','').replace('_','-') for title in titles] 
    temp = ''
    for t in titles: 
        temp += (t+',')
    temp = temp.strip(',')
    if len(titles)>6: 
        temp = destination_folder.replace('results/', '')

    #plt.figure(figsize=(20, 15), dpi=100)
    plt.figure(dpi=80, figsize=(5, 4))
    plt.rcParams.update({'font.size':20})
    results_counts.plot.area(stacked=True, ylabel = '# action pairs obs.', #rot=45,
        xlabel='Iteration (2 players at any one time)', #colormap='PiYG_r',
        color={'C, C':'#28641E', 'C,D/D,C':'#FFFFFF', 'D, D':'#8E0B52'}, linewidth=0.05, alpha=0.9,
        #color={'C, C':'#28641E', 'C, D':'#B0DC82', 'D, C':'#EEAED4', 'D, D':'#8E0B52'}, linewidth=0.05, alpha=0.9,
        #color={'C, C':'royalblue', 'C, D':'lightblue', 'D, C':'yellow', 'D, D':'orange'},
        title = str('Population: ' + temp)) #Pairs of simultaneous actions over time: \n '+

    #plt.savefig(f'{destination_folder}/plots/action_pairs.png', bbox_inches='tight')
    if not os.path.isdir('results/outcome_plots/actions'):
            os.makedirs('results/outcome_plots/actions')

    if not option: #if plotting main results
        population = destination_folder.split('/')[1]

    else: #if plotting extra parameter search for beta in QLVM 
        population = destination_folder
        #pair += str(option)
    
    plt.savefig(f'results/outcome_plots/actions/popul_{population}.pdf', bbox_inches='tight')

def plot_action_pairs_reduced_population(destination_folder, titles, n_runs, option=None):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    if False: #if os.path.exists(str(destination_folder+'/action_pairs.csv')):
        results = pd.read_csv(str(destination_folder+'/action_pairs.csv'), index_col=0)

    else: #if action_pairs.csv does not yet exist 
        actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)
        actions_player2 = pd.read_csv(f'{destination_folder}/player2/action.csv', index_col=0)

        #rename columns 
        colnames = ['run'+str(i) for i in range(n_runs)]
        actions_player1.columns = colnames
        actions_player2.columns = colnames

        results = pd.DataFrame(columns=colnames)

        for colname in colnames: #loop over every run
            str_value = actions_player1[colname].astype(str) + ', ' + actions_player2[colname].astype(str)
            str_value = str_value.str.replace('1', 'D')
            str_value = str_value.str.replace('0', 'C')
            str_value = str_value.replace('C, D', 'C,D/D,C').replace('D, C', 'C,D/D,C') #the order of actions does not matter, so comibe CD/DC into one 
            results[colname] = str_value

        results.to_csv(str(destination_folder+'/action_pairs.csv'))

    #keep every 100th row 
    new_results = results[results.index % 20 == 0]

    results_counts = new_results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first episode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    titles = [title.replace('QL','') for title in titles] 
    titles = [title.replace('Ethics','').replace('_','-') for title in titles] 
    temp = ''
    for t in titles: 
        temp += (t+',')
    temp = temp.strip(',')
    if len(titles)>6: 
        temp = destination_folder.replace('results/', '')

    #plt.figure(figsize=(20, 15), dpi=100)
    plt.figure(dpi=80, figsize=(5, 4))
    plt.rcParams.update({'font.size':20})
    results_counts.plot.area(stacked=True, ylabel = '# action pairs obs.', #rot=45,
        xlabel='Iteration (2 players at any one time)', #colormap='PiYG_r',
        color={'C, C':'#28641E', 'C,D/D,C':'#FFFFFF', 'D, D':'#8E0B52'}, linewidth=0.05, alpha=0.9,
        #color={'C, C':'#28641E', 'C, D':'#B0DC82', 'D, C':'#EEAED4', 'D, D':'#8E0B52'}, linewidth=0.05, alpha=0.9,
        #color={'C, C':'royalblue', 'C, D':'lightblue', 'D, C':'yellow', 'D, D':'orange'},
        title = str('Population: ' + temp)) #Pairs of simultaneous actions over time: \n '+

    #plt.savefig(f'{destination_folder}/plots/action_pairs.png', bbox_inches='tight')
    if not os.path.isdir('results/outcome_plots/actions'):
            os.makedirs('results/outcome_plots/actions')

    if not option: #if plotting main results
        population = destination_folder.split('/')[1]

    else: #if plotting extra parameter search for beta in QLVM 
        population = destination_folder
        #pair += str(option)
    
    plt.savefig(f'results/outcome_plots/actions/popul_{population}_reduced.pdf', bbox_inches='tight')

def plot_action_types_area_population(destination_folder, titles, n_runs):
    '''visualise action types that each individual player takes against their opponent's last move 
    --> what strategies are being learnt at all steps of the run? 
    - consider the whole run'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    titles = [title.replace('QL','') for title in titles] 

    temp = ''
    for t in titles: 
        temp += (t+',')
    temp = temp.strip(',')

    ##############################################
    ### Plot from the perspective of player1 first ####

    actions_player1 = pd.read_csv(f'{destination_folder}/player1/action.csv', index_col=0)
    state_player1 = pd.read_csv(f'{destination_folder}/player1/state.csv', index_col=0)
    #rename columns 
    colnames = ['run'+str(i) for i in range(n_runs)]
    actions_player1.columns = colnames
    state_player1.columns = colnames
    results = pd.DataFrame(columns=colnames)

    for colname in colnames: 
        str_value = actions_player1[colname].astype(str) + ' | ' + state_player1[colname].astype(str)
        str_value = str_value.str.replace('1', 'D')
        str_value = str_value.str.replace('0', 'C')
        results[colname] = str_value

    #results.to_csv(str(destination_folder+'/player1/action_types_full.csv'))
    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first apisode they are reacting to default state=0
    results_counts.dropna(axis=1, how='all', inplace=True)

    #plt.figure(figsize=(40, 25), dpi=100)
    results_counts.plot.area(stacked=True, ylabel = '# agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
        xlabel='Iteration', colormap='PiYG_r',
        #color={'C | (C,C)':'lightblue', 'C | (C,D)':'deepskyblue', 'C | (D,C)':'cyan', 'C | (D,D)':'royalblue', 'D | (C,C)':'orange', 'D | (C,D)':'yellow', 'D | (D,C)':'peru', 'D | (D,D)':'dimgrey'}, 
        title='Types of actions over time - player1: \n '+str(temp))
    plt.legend(loc=8, fontsize=13)
    plt.savefig(f'{destination_folder}/plots/action_types_area_player1.png', bbox_inches='tight')


    ##############################################
    ### Plot from the perspective of player2 ####

    actions_player2 = pd.read_csv(f'{destination_folder}/player2/action.csv', index_col=0)
    state_player2 = pd.read_csv(f'{destination_folder}/player2/state.csv', index_col=0)
    #rename columns - use colnames form above
    actions_player2.columns = colnames
    state_player2.columns = colnames
    results = pd.DataFrame(columns=colnames)

    for colname in colnames: 
        str_value = actions_player2[colname].astype(str) + ' | ' + state_player2[colname].astype(str)
        str_value = str_value.str.replace('1', 'D')
        str_value = str_value.str.replace('0', 'C')
        results[colname] = str_value

    #results.to_csv(str(destination_folder+'/player2/action_types_full.csv'))
    results_counts = results.transpose().apply(pd.value_counts).transpose()[1:]
    results_counts.dropna(axis=1, how='all', inplace=True)

    #plt.figure(figsize=(40, 15), dpi=100)
    results_counts.plot.area(stacked=True, ylabel = '# agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
        xlabel='Iteration', colormap='PiYG_r',
        #color={'C | C':'lightblue', 'C | D':'yellow', 'D | C':'brown', 'D | D':'orange'},
        #color={'C | (C,C)':'lightblue', 'C | (C,D)':'yellow', 'D | (D,C)':'brown', 'D | (D,D)':'orange',
        #        'D | (C,C)':'darkblue', 'D | (C,D)':'mustard', 'C | (D,C)':'grey', 'C | (D,D)':'red'}, 
        title='Types of actions over time - player2: \n '+str(temp))
    plt.legend(loc=8, fontsize=13)
    plt.savefig(f'{destination_folder}/plots/action_types_area_player2.png', bbox_inches='tight')



def plot_last_20_actions_eachplayerinpopulation(destination_folder, titles, n_runs):
    '''deep-dive into the strategy learnt by the end - interpert the last 20 moves'''
    #TO DO fix colormap to be fixed by value
    ##############################################
    ### Plot from the perspective of player1 first ####
    for idx in range(len(titles)):
        title = title_mapping[titles[idx]]
        actions_given_states = pd.read_csv(f'{destination_folder}/player_action_types/{title}_{idx}.csv', index_col=0).sort_index()[-20:]

        results_counts = actions_given_states.transpose().apply(pd.value_counts).transpose()[1:] 
        #plot after first episode, as in the first apisode they are reacting to default state=0
        #results_counts.dropna(axis=1, how='all', inplace=True)
        #results_counts = results_counts.reset_index().drop('index', axis=1)
        results_counts_clean = results_counts.divide(results_counts.sum(axis=1), axis=0).multiply(100)

        #remove keys from color which are not in results_counts_clean.columns
        #newcolor = {k: color[k] for k in results_counts_clean.columns}

        plt.figure(figsize=(20, 15), dpi=100)
        results_counts_clean.plot.bar(stacked=True, ylabel = '# agents taking this type of action \n (across '+str(n_runs)+' runs)', rot=45,
            xlabel='Iteration', 
            #colormap='PiYG_r',
            #color={'C | C':'lightblue', 'C | D':'yellow', 'D | C':'brown', 'D | D':'orange'}, 
            color={'C | (C, C)':'#28641E', 'C | (C, D)':'#63A336', 'C | (D, C)':'#B0DC82', 'C | (D, D)':'#EBF6DC', 'D | (C, C)':'#FBE6F1', 'D | (C, D)':'#EEAED4', 'D | (D, C)':'#CE4591', 'D | (D, D)':'#8E0B52'}, 
            title='Last 20 action types: \n '+title+' agent '+str(idx))
        plt.savefig(f'{destination_folder}/plots/last_20_actions_{title}_{idx}.png', bbox_inches='tight')

def visualise_last_20_actions_matrix_eachplayerinpopulation(destination_folder):
    '''explore individual strategies learnt by individual players (not a collection of 100 players) 
    - look at 20 last moves as a vector'''

    for idx in range(len(titles)):
        title = title_mapping[titles[idx]]
        actions_given_states = pd.read_csv(f'{destination_folder}/player_action_types/{title}_{idx}.csv', index_col=0).sort_index()[-20:]
        actions_given_states = actions_given_states.transpose()

        #caption = destination_folder.replace('results/', '')
        #caption = caption.replace('QL', '')
        actions_given_states = actions_given_states.style.applymap(C_condition).set_caption(f"Player {title}_{idx}") #from {popuilation}
        dfi.export(actions_given_states,f"{destination_folder}/plots/table_export_{title}_{idx}_last20.png")

def plot_historgram_interactions(run_idx, num_iter, population_size, plot_selectOR=False):
    temp = pd.read_csv(destination_folder+f'/history/run{run_idx}.csv', index_col = 0)
    #temp['title_p1'].fillna('Selfish', inplace=True)
    #temp['title_p2'].fillna('Selfish', inplace=True)

    if plot_selectOR:
        plt.figure(figsize=(14,5))
        plt.hist(temp['idx_p1'], edgecolor='black', linewidth=1.2)#, bins=50)
        plt.title(f'Histogram of players acting as leader (selectOR) \n run {run_idx}')
        plt.xlabel('player idx')
        plt.xticks(rotation=90)
        plt.ylabel(f'Count \n (over {num_iter} iterations)')
        plt.xticks(list(range(population_size)))
        plt.show()
        plt.savefig(f'{destination_folder}/plots/hist_interactions_slector_run{run_idx}.pdf', bbox_inches='tight')

    plt.figure(figsize=(14,5))
    plt.hist(temp['idx_p2'], edgecolor='black', linewidth=1.2)#, bins=50)
    plt.title(f'Histogram of players acting as the opponent (selectED) \n run {run_idx}')
    plt.xlabel('player idx')
    plt.xticks(rotation=90)
    plt.ylabel(f'Count \n (over {num_iter} iterations)')
    plt.xticks(list(range(population_size)))
    plt.show()
    plt.savefig(f'{destination_folder}/plots/hist_interactions_selected_run{run_idx}.pdf', bbox_inches='tight')




def plot_one_run_Q_values_population(Q_values_list, run_idx, player_idx):
    '''plot progression of Q-value updates over the 10000 iterations for one example run - separately for each state.
    We plot both actions (C and D) on one plot to copare which action ends up being optimal in every state.'''
    rXs0a0_list = [iteration[0,0] for iteration in Q_values_list] #run x, state0, action1
    rXs0a1_list = [iteration[0,1] for iteration in Q_values_list]
    rXs1a0_list = [iteration[1,0] for iteration in Q_values_list]
    rXs1a1_list = [iteration[1,1] for iteration in Q_values_list]
    rXs2a0_list = [iteration[2,0] for iteration in Q_values_list]
    rXs2a1_list = [iteration[2,1] for iteration in Q_values_list]
    rXs3a0_list = [iteration[3,0] for iteration in Q_values_list]
    rXs3a1_list = [iteration[3,1] for iteration in Q_values_list]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    #fig.suptitle(f'Q-values for all state-action pairs, run{run_idx} \n')
    #fig.suptitle(str(f'run{run_idx}, player{player_idx}'+'\n'+'\n'))
    ax1.plot(range(len(rXs0a0_list)), rXs0a0_list[:], label=f'Action=C', color='darkgreen')
    ax1.plot(range(len(rXs0a1_list)), rXs0a1_list[:], label=f'Action=D', color='purple')
    ax1.set_title('State S=(C,C)')
    ax2.plot(range(len(rXs1a0_list)), rXs1a0_list[:], label=f'Action=C', color='darkgreen')
    ax2.plot(range(len(rXs1a1_list)), rXs1a1_list[:], label=f'Action=D', color='purple')
    ax2.set_title('S=(C,D)')
    ax3.plot(range(len(rXs2a0_list)), rXs2a0_list[:], label=f'Action=C', color='darkgreen')
    ax3.plot(range(len(rXs2a1_list)), rXs2a1_list[:], label=f'Action=D', color='purple')
    ax3.set_title('S=(D,C)')
    ax4.plot(range(len(rXs3a0_list)), rXs3a0_list[:], label=f'Action=C', color='darkgreen')
    ax4.plot(range(len(rXs3a1_list)), rXs3a1_list[:], label=f'Action=D', color='purple')
    ax4.set_title('S=(D,D)')
    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    ax1.set(ylabel='Q-value')
    plt.suptitle(str(f'run{run_idx}, player{player_idx}'+'\n'))

    #max_Q_value = max(max(rXs0a0_list), max(rXs0a1_list), max(rXs1a0_list), max(rXs1a1_list)) #max q-value on this run
    #plt.ylim(0, int(max_Q_value+5))
    #set y axis limit based on the claculation of a max possible return in this game and the discount factor: r=4, gamma=0.9, max return = r*(1/1-gamma)
    
    #if game_title == 'IPD': 
    #    plt.ylim(0, 40+5) 
    #elif game_title == 'VOLUNTEER':
    #    plt.ylim(0, 50+5)
    #elif game_title == 'STAGHUNT':
    #    plt.ylim(0, 50+5)
    #then export the interactive outupt / single plot as pdf/html to store the results compactly 

def plot_one_run_Q_values_selection_population(Q_values_list, run_idx, player_idx):
    '''plot progression of selection Q-value updates over the 10000 iterations for one example run - separately for each selection state.
    We plot all selections (population size - 1) on one plot to compare which action ends up being optimal in every state.'''
    
    '''NOTE this is a custom function for population size 3 at the moment!! '''
    state_size = 2 #TO DO make these customisable from config
    action_size = 2 #TO DO make these customisable from config
    population_size = 3 #TO DO make these customisable from config
    num_possible_selection_states = (state_size*action_size)**(population_size-1) #16
    #to do make the below customisable so there are 2*num_possible_selection_states dataframes 

    rXs0a0_list = [iteration[0,0] for iteration in Q_values_list] #run x, state0, action1
    rXs0a1_list = [iteration[0,1] for iteration in Q_values_list]
    rXs1a0_list = [iteration[1,0] for iteration in Q_values_list]
    rXs1a1_list = [iteration[1,1] for iteration in Q_values_list]
    rXs2a0_list = [iteration[2,0] for iteration in Q_values_list]
    rXs2a1_list = [iteration[2,1] for iteration in Q_values_list]
    rXs3a0_list = [iteration[3,0] for iteration in Q_values_list]
    rXs3a1_list = [iteration[3,1] for iteration in Q_values_list]
    rXs4a0_list = [iteration[4,0] for iteration in Q_values_list]
    rXs4a1_list = [iteration[4,1] for iteration in Q_values_list]
    rXs5a0_list = [iteration[5,0] for iteration in Q_values_list]
    rXs5a1_list = [iteration[5,1] for iteration in Q_values_list]
    rXs6a0_list = [iteration[6,0] for iteration in Q_values_list]
    rXs6a1_list = [iteration[6,1] for iteration in Q_values_list]
    rXs7a0_list = [iteration[7,0] for iteration in Q_values_list]
    rXs7a1_list = [iteration[7,1] for iteration in Q_values_list]
    rXs8a0_list = [iteration[8,0] for iteration in Q_values_list]
    rXs8a1_list = [iteration[8,1] for iteration in Q_values_list]
    rXs9a0_list = [iteration[9,0] for iteration in Q_values_list]
    rXs9a1_list = [iteration[9,1] for iteration in Q_values_list]
    rXs10a0_list = [iteration[10,0] for iteration in Q_values_list]
    rXs10a1_list = [iteration[10,1] for iteration in Q_values_list]
    rXs11a0_list = [iteration[11,0] for iteration in Q_values_list]
    rXs11a1_list = [iteration[11,1] for iteration in Q_values_list]
    rXs12a0_list = [iteration[12,0] for iteration in Q_values_list]
    rXs12a1_list = [iteration[12,1] for iteration in Q_values_list]
    rXs13a0_list = [iteration[13,0] for iteration in Q_values_list]
    rXs13a1_list = [iteration[13,1] for iteration in Q_values_list]
    rXs14a0_list = [iteration[14,0] for iteration in Q_values_list]
    rXs14a1_list = [iteration[14,1] for iteration in Q_values_list]
    rXs15a0_list = [iteration[15,0] for iteration in Q_values_list]
    rXs15a1_list = [iteration[15,1] for iteration in Q_values_list]

    if True: 
        #drop nans from all lists to help with plotting 
        for s_a_list in [rXs0a0_list, rXs0a1_list, rXs1a0_list, rXs1a1_list,
                        rXs2a0_list, rXs2a1_list, rXs3a0_list, rXs3a1_list,
                        rXs4a0_list, rXs4a1_list, rXs5a0_list, rXs5a1_list,
                        rXs6a0_list, rXs6a1_list, rXs7a0_list, rXs7a1_list,
                        rXs8a0_list, rXs8a1_list, rXs9a0_list, rXs9a1_list,
                        rXs10a0_list, rXs10a1_list, rXs11a0_list, rXs11a1_list,
                        rXs12a0_list, rXs12a1_list, rXs13a0_list, rXs13a1_list,
                        rXs14a0_list, rXs14a1_list, rXs15a0_list, rXs15a1_list]: 
            s_a_list[:] = [x for x in s_a_list if str(x) != 'nan']


    fig, (
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16) = plt.subplots(
        1, 16, figsize=(40, 4), sharey=True)
    #fig.suptitle(f'Q-values for all state-action pairs, run{run_idx} \n')
    fig.suptitle(str(f'run{run_idx}, player{player_idx}'), x=1)
#    fig.suptitle(str(f'run{run_idx}, player{player_idx}'+'\n'+'\n'))

    #opponent1, color1 = 'AC', 'darkgreen'
    #opponent2, color2 = 'AD', 'purple
    opponent1, color1 = 'S1', 'purple'
    opponent2, color2 = 'S2', 'cyan'

    ax1.plot(range(len(rXs0a0_list)), rXs0a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax1.plot(range(len(rXs0a1_list)), rXs0a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax1.set_title('State=CC,CC')
    ax2.plot(range(len(rXs1a0_list)), rXs1a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax2.plot(range(len(rXs1a1_list)), rXs1a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax2.set_title('S=CC,CD')
    ax3.plot(range(len(rXs2a0_list)), rXs2a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax3.plot(range(len(rXs2a1_list)), rXs2a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax3.set_title('S=CC,DC')
    ax4.plot(range(len(rXs3a0_list)), rXs3a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax4.plot(range(len(rXs3a1_list)), rXs3a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax4.set_title('S=CC,DD')

    ax5.plot(range(len(rXs4a0_list)), rXs4a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax5.plot(range(len(rXs4a1_list)), rXs4a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax5.set_title('S=CD,CC')
    ax6.plot(range(len(rXs5a0_list)), rXs5a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax6.plot(range(len(rXs5a1_list)), rXs5a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax6.set_title('S=CD,CD')
    ax7.plot(range(len(rXs6a0_list)), rXs6a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax7.plot(range(len(rXs6a1_list)), rXs6a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax7.set_title('S=CD,DC')
    ax8.plot(range(len(rXs7a0_list)), rXs7a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax8.plot(range(len(rXs7a1_list)), rXs7a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax8.set_title('S=CD,DD')

    ax9.plot(range(len(rXs8a0_list)), rXs8a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax9.plot(range(len(rXs8a1_list)), rXs8a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax9.set_title('S=DC,CC')
    ax10.plot(range(len(rXs9a0_list)), rXs9a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax10.plot(range(len(rXs9a1_list)), rXs9a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax10.set_title('S=DC,CD')
    ax11.plot(range(len(rXs10a0_list)), rXs10a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax11.plot(range(len(rXs10a1_list)), rXs10a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax11.set_title('S=DC,DC')
    ax12.plot(range(len(rXs11a0_list)), rXs11a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax12.plot(range(len(rXs11a1_list)), rXs11a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax12.set_title('S=DC,DD')

    ax13.plot(range(len(rXs12a0_list)), rXs12a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax13.plot(range(len(rXs12a1_list)), rXs12a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax13.set_title('S=DD,CC')
    ax14.plot(range(len(rXs13a0_list)), rXs13a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax14.plot(range(len(rXs13a1_list)), rXs13a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax14.set_title('S=DD,CD')
    ax15.plot(range(len(rXs14a0_list)), rXs14a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax15.plot(range(len(rXs14a1_list)), rXs14a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax15.set_title('S=DD,DC')
    ax16.plot(range(len(rXs15a0_list)), rXs15a0_list[:], label=f'Action=choose {opponent1}', color=color1)
    ax16.plot(range(len(rXs15a1_list)), rXs15a1_list[:], label=f'Action=choose {opponent2}', color=color2)
    ax16.set_title('S=DD,DD')



    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    ax1.set(ylabel='Q-value')


def plot_Q_values_population(destination_folder, population_size, n_runs, which, record_Qvalues, fewer_than_nruns = False): #NB set manually which Q values are being plotted
    if fewer_than_nruns == True:
        n_runs = 10 

    for player_idx in range(population_size):
            
        for run_idx in range(n_runs): 
            run_idx += 1 

            if which == 'local':
                history_Qvalues_local_player = np.load(f'{destination_folder}/QVALUES/Q_VALUES_local_player{player_idx}_list_run{run_idx}.npy', allow_pickle=True)
                print(f'Printing local Q-values, player{player_idx}, run{run_idx}:')
                if record_Qvalues == 'dilemma':
                    plot_one_run_Q_values_population(Q_values_list = history_Qvalues_local_player, run_idx = run_idx, player_idx = player_idx)
        #            plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_local_player, run_idx = run_idx)
                if record_Qvalues == 'selection':
                    plot_one_run_Q_values_selection_population(Q_values_list = history_Qvalues_local_player, run_idx = run_idx, player_idx = player_idx)


            elif which == 'target':
                try: 
                    print(f'Printing target Q-values, player{player_idx}, , run{run_idx}:')
                    history_Qvalues_target_player = np.load(f'{destination_folder}/QVALUES/Q_VALUES_target_{player_idx}_list_run{run_idx}.npy', allow_pickle=True)
                    plot_one_run_Q_values_population(Q_values_list = history_Qvalues_target_player, run_idx = run_idx, player_idx = player_idx)
        #                plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_target_player, run_idx = run_idx)
                except:
                    print(f'Unable to locate target Q-values, player{player_idx}')


def plot_Loss_population(destination_folder, population_size, n_runs, fewer_than_nruns = False):
    if fewer_than_nruns == True:
        n_runs = 10 

    #plot for player_x
    for player_idx in range(population_size):
            
        #run_idx = 0
        for run_idx in range(n_runs): 
            run_idx += 1 

            #try: 
            LOSS_player = np.load(f'{destination_folder}/LOSS/LOSS_player{player_idx}_list_run{run_idx}.npy', allow_pickle=True)
            #print(f'Printing Loss, player{player_idx}:')
#            loss_list = [i if i else float(0) for i in LOSS_player[run_idx-1]] #replace None with 0 
            loss_list = [i if i else np.nan for i in LOSS_player[0]] #replace None with 0 

#            loss_list = [float(i) if i!='<0.0001' else float(0) for i in loss_list] #replace all str '<0.001' with 0
            loss_list = [i if i!='<0.0001' else float(0) for i in loss_list] #replace all str '<0.001' with 0
            plot_one_run_Loss_population(LOSS_list = loss_list, run_idx = run_idx, player_idx=player_idx)
            #except:
            #    print(f'Unable to locate Loss, player{player_idx}')


def plot_Loss_population_subplots(destination_folder, population_size, n_runs):
    for player_idx in range(population_size):
            
        #run_idx = 0
        for run_idx in range(n_runs): 
            run_idx += 1 

            LOSS_player = np.load(f'{destination_folder}/LOSS/LOSS_player{player_idx}_list_run{run_idx}.npy', allow_pickle=True)
            #print(f'Printing Loss, player{player_idx}:')
#            loss_list = [i if i else float(0) for i in LOSS_player[run_idx-1]] #replace None with 0 
            loss_list = [i if i else np.nan for i in LOSS_player[0]] #replace None with 0 

#            loss_list = [float(i) if i!='<0.0001' else float(0) for i in loss_list] #replace all str '<0.001' with 0
            loss_list = [i if i!='<0.0001' else float(0) for i in loss_list] #replace all str '<0.001' with 0
            plot_one_run_Loss_population(LOSS_list = loss_list, run_idx = run_idx, player_idx=player_idx)

def plot_Loss_population_new_onerun(run_idx, destination_folder, population_size, record_Loss):

    fig, axs = plt.subplots(1, population_size, figsize=(3 * population_size, 6), sharey=True)
    #fig.subplots_adjust(hspace=0.5)

    for player_idx in range(population_size):
        LOSS_player = np.load(f'{destination_folder}/LOSS/LOSS_player{player_idx}_list_run{run_idx}.npy', allow_pickle=True)
        loss_list = [i if i else np.nan for i in LOSS_player[0]]
        loss_list = [i if i != '<0.0001' else float(0) for i in loss_list]

        ax = axs[player_idx]
        ax.plot(loss_list, linewidth=0.7) #label='loss for DQN Player', 
        ax.set_xticklabels(ax.get_xticks(), rotation = 90)
        #ax.plot(loss_list[~np.isnan(loss_list)], label='loss for DQN Player', linewidth=0.1)
        #if min(loss_list):
        #    if max(loss_list):
        #        ax.set_ylim(min(loss_list) - 1, max(loss_list) + 1)
        #ax.set_ylim(min(loss_list[~np.isnan(loss_list)]) - 1, max(loss_list[~np.isnan(loss_list)]) + 1)

        ax.set_title(f'Player{player_idx}')

    plt.suptitle(f'{record_Loss} Loss over time, Run{run_idx}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()


########################
#### group plots ####
########################

def plot_relative_reward(player1_title, n_runs):
    '''plot game reward - relatie cumulative (bar & over time) & per iteration
    - how well off did the players end up relative to each other on the game?'''
    ##################################
    #### bar chart game cumulative reward for player1_tytle vs others  ####
    ##################################
    against_QLS = pd.read_csv(f'results/{player1_title}_QLS/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    against_QLS_means = against_QLS.mean()
    against_QLS_sds = against_QLS.std()
    against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

    try:
        against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    except:
        against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    against_QLUT_means = against_QLUT.mean()
    against_QLUT_sds = against_QLUT.std()
    against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

    try:
        against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    except:
        against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    against_QLDE_means = against_QLDE.mean()
    against_QLDE_sds = against_QLDE.std()
    against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

    try:
        against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    except:
        against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    against_QLVE_e_means = against_QLVE_e.mean()
    against_QLVE_e_sds = against_QLVE_e.std()
    against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

    try:
        against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    except:
        against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    against_QLVE_k_means = against_QLVE_k.mean()
    against_QLVE_k_sds = against_QLVE_k.std()
    against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

    try:
        against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    except:
        against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
    against_QLVM_means = against_QLVM.mean()
    against_QLVM_sds = against_QLVM.std()
    against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)


    #import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3, 3), dpi=80)
    plt.rcParams.update({'font.size':20})
    ax = fig.add_axes([0,0,1,1])
    #labels = ['vs_S', 'vs_UT', 'vs_DE', 'vs_'+r'$VE_e$', 'vs_'+r'$VE_k$'] 
    labels = ['S', 'UT', 'DE', 'VE'+r'$_e$', 'VE'+r'$_k$'] 
    means = [against_QLS_means,against_QLUT_means,against_QLDE_means,against_QLVE_e_means,against_QLVE_k_means]
    cis = [against_QLS_ci,against_QLUT_ci,against_QLDE_ci,against_QLVE_e_ci, against_QLVE_k_ci]
    colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple']  
    #add QLVM mixed agent 
    labels.append('VE'+r'$_m$')
    means.append(against_QLVM_means)
    cis.append(against_QLVM_ci)
    colors.append('pink')
    ax.bar(labels, means, yerr=cis, color=colors, width = 0.8) #capsize=7, 
    #plt.xticks(rotation=45)
    ax.set_ylabel(r'Cumulative $R_{extr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
    #(f'R^{extr}_{i=QLUT}'# for {player1_title}')
    long_title = title_mapping[player1_title].replace('Ethics','').replace('_', '-')
    ax.set_title(long_title +' vs other \n') #'Cumulative Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
    ax.set_xlabel('Opponent type')
    if game_title=='IPD': #NOTE game_title is set outside this function - in the overall environment - see code below 
        plt.gca().set_ylim([0, 40000])
    elif game_title=='VOLUNTEER':
        plt.gca().set_ylim([0, 50000])
    elif game_title=='STAGHUNT': 
        plt.gca().set_ylim([0, 50000])  
    if not os.path.isdir('results/outcome_plots/reward'):
        os.makedirs('results/outcome_plots/reward')
    plt.savefig(f'results/outcome_plots/reward/bar_cumulative_game_reward_{player1_title}.pdf', bbox_inches='tight')
    
    ##################################
    #### cumulative - game reward for player1_tytle vs others  ####
    ##################################
    against_QLS = pd.read_csv(f'results/{player1_title}_QLS/player1/df_cumulative_reward_game.csv', index_col=0)
    against_QLS_means = against_QLS.mean(axis=1)
    against_QLS_sds = against_QLS.std(axis=1)
    against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

    try:
        against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/player1/df_cumulative_reward_game.csv', index_col=0)
    except:
        against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0)
    against_QLUT_means = against_QLUT.mean(axis=1)
    against_QLUT_sds = against_QLUT.std(axis=1)
    against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

    try:
        against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/player1/df_cumulative_reward_game.csv', index_col=0)
    except:
        against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0)
    against_QLDE_means = against_QLDE.mean(axis=1)
    against_QLDE_sds = against_QLDE.std(axis=1)
    against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

    try:
        against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/player1/df_cumulative_reward_game.csv', index_col=0)
    except:
        against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0)
    against_QLVE_e_means = against_QLVE_e.mean(axis=1)
    against_QLVE_e_sds = against_QLVE_e.std(axis=1)
    against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

    try:
        against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/player1/df_cumulative_reward_game.csv', index_col=0)
    except:
        against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0)
    against_QLVE_k_means = against_QLVE_k.mean(axis=1)
    against_QLVE_k_sds = against_QLVE_k.std(axis=1)
    against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

    try:
        against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/player1/df_cumulative_reward_game.csv', index_col=0)
    except:
        against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0)
    against_QLVM_means = against_QLVM.mean(axis=1)
    against_QLVM_sds = against_QLVM.std(axis=1)
    against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)

    plt.figure(dpi=80) #figsize=(10, 6)
    plt.plot(against_QLS.index[:], against_QLS_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLS', color='red')
    plt.fill_between(against_QLS.index[:], against_QLS_means-against_QLS_ci, against_QLS_means+against_QLS_ci, facecolor='#ff9999', alpha=0.5)
    plt.plot(against_QLUT.index[:], against_QLUT_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLUT', color='#556b2f')
    plt.fill_between(against_QLUT.index[:], against_QLUT_means-against_QLUT_ci, against_QLUT_means+against_QLUT_ci, facecolor='#ccff99', alpha=0.5)
    plt.plot(against_QLDE.index[:], against_QLDE_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLDE', color='#00cccc')
    plt.fill_between(against_QLDE.index[:], against_QLDE_means-against_QLDE_ci, against_QLDE_means+against_QLDE_ci, facecolor='#99ffff', alpha=0.5)
    plt.plot(against_QLVE_e.index[:], against_QLVE_e_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLVE_e', color='orange')
    plt.fill_between(against_QLVE_e.index[:], against_QLVE_e_means-against_QLVE_e_ci, against_QLVE_e_means+against_QLVE_e_ci, facecolor='#ffcc99', alpha=0.5)
    plt.plot(against_QLVE_k.index[:], against_QLVE_k_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLVE_k', color='purple')
    plt.fill_between(against_QLVE_k.index[:], against_QLVE_k_means-against_QLVE_k_ci, against_QLVE_k_means+against_QLVE_k_ci, facecolor='#CBC3E3', alpha=0.5)
    plt.plot(against_QLVM.index[:], against_QLVM_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLVM', color='pink')
    plt.fill_between(against_QLVM.index[:], against_QLVM_means-against_QLVM_ci, against_QLVM_means+against_QLVM_ci, facecolor='pink', alpha=0.5)
    

    #long_title = title_mapping[player1_title]
    plt.title(long_title +' vs other \n') #r'Cumulative game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
    if game_title=='IPD':
        plt.gca().set_ylim([0, 40000])
    elif game_title=='VOLUNTEER':
        plt.gca().set_ylim([0, 50000])
    elif game_title=='STAGHUNT': 
        plt.gca().set_ylim([0, 50000])  
    plt.ylabel(r'Cumulative $R_{extr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
    plt.xlabel('Iteration')
    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    plt.savefig(f'results/outcome_plots/reward/cumulative_game_reward_{player1_title}.pdf', bbox_inches='tight')


    ##################################
    #### non-cumulative - game reward for player1_tytle vs others  ####
    ##################################
    figsize=(5, 4)

    against_QLS = pd.read_csv(f'results/{player1_title}_QLS/player1/df_reward_game.csv', index_col=0)
    against_QLS_means = against_QLS.mean(axis=1)
    against_QLS_sds = against_QLS.std(axis=1)
    against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

    try:
        against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/player1/df_reward_game.csv', index_col=0)
    except:
        against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/player2/df_reward_game.csv', index_col=0)
    against_QLUT_means = against_QLUT.mean(axis=1)
    against_QLUT_sds = against_QLUT.std(axis=1)
    against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

    try:
        against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/player1/df_reward_game.csv', index_col=0)
    except:
        against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/player2/df_reward_game.csv', index_col=0)
    against_QLDE_means = against_QLDE.mean(axis=1)
    against_QLDE_sds = against_QLDE.std(axis=1)
    against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

    try:
        against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/player1/df_reward_game.csv', index_col=0)
    except:
        against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/player2/df_reward_game.csv', index_col=0)
    against_QLVE_e_means = against_QLVE_e.mean(axis=1)
    against_QLVE_e_sds = against_QLVE_e.std(axis=1)
    against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

    try:
        against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/player1/df_reward_game.csv', index_col=0)
    except:
        against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/player2/df_reward_game.csv', index_col=0)
    against_QLVE_k_means = against_QLVE_k.mean(axis=1)
    against_QLVE_k_sds = against_QLVE_k.std(axis=1)
    against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

    try:
        against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/player1/df_reward_game.csv', index_col=0)
    except:
        against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/player2/df_reward_game.csv', index_col=0)
    against_QLVM_means = against_QLVM.mean(axis=1)
    against_QLVM_sds = against_QLVM.std(axis=1)
    against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)

    plt.figure(dpi=80, figsize=figsize)
    plt.rcParams.update({'font.size':20})
    plt.plot(against_QLS.index[:], against_QLS_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLS', color='red')
    plt.fill_between(against_QLS.index[:], against_QLS_means-against_QLS_ci, against_QLS_means+against_QLS_ci, facecolor='#ff9999', alpha=0.5)
    plt.plot(against_QLUT.index[:], against_QLUT_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLUT', color='#556b2f')
    plt.fill_between(against_QLUT.index[:], against_QLUT_means-against_QLUT_ci, against_QLUT_means+against_QLUT_ci, facecolor='#ccff99', alpha=0.5)
    plt.plot(against_QLDE.index[:], against_QLDE_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLDE', color='#00cccc')
    plt.fill_between(against_QLDE.index[:], against_QLDE_means-against_QLDE_ci, against_QLDE_means+against_QLDE_ci, facecolor='#99ffff', alpha=0.5)
    plt.plot(against_QLVE_e.index[:], against_QLVE_e_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLVE_e', color='orange')
    plt.fill_between(against_QLVE_e.index[:], against_QLVE_e_means-against_QLVE_e_ci, against_QLVE_e_means+against_QLVE_e_ci, facecolor='#ffcc99', alpha=0.5)
    plt.plot(against_QLVE_k.index[:], against_QLVE_k_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLVE_k', color='purple')
    plt.fill_between(against_QLVE_k.index[:], against_QLVE_k_means-against_QLVE_k_ci, against_QLVE_k_means+against_QLVE_k_ci, facecolor='#CBC3E3', alpha=0.5)
    plt.plot(against_QLVM.index[:], against_QLVM_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLVM', color='palevioletred')
    plt.fill_between(against_QLVM.index[:], against_QLVM_means-against_QLVM_ci, against_QLVM_means+against_QLVM_ci, facecolor='pink', alpha=0.5)
    

    #long_title = title_mapping[player1_title]
    plt.title(long_title +' vs other \n') #r'Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
    if game_title=='IPD':
        plt.gca().set_ylim([1, 4])
    elif game_title=='VOLUNTEER':
        plt.gca().set_ylim([1, 5])
    elif game_title=='STAGHUNT': 
        plt.gca().set_ylim([1, 5])
    plt.ylabel(r'$R_{extr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
    plt.xlabel('Iteration')
    #plt.xticks(rotation=45)
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)
    plt.savefig(f'results/outcome_plots/reward/game_reward_{player1_title}.pdf', bbox_inches='tight')

def plot_relative_moral_reward(player1_title, n_runs):
    '''plot moral reward - relatie cumulative (over time & bar plot) & per iteration
    - how well off did the players end up relative to each other in terms of moral reward?'''
    
    ##################################
    #### moral cumulative reward for player1_tytle vs others  ####
    ##################################
    against_QLS = pd.read_csv(f'results/{player1_title}_QLS/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    against_QLS_means = against_QLS.mean()
    against_QLS_sds = against_QLS.std()
    against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

    try:
        against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    except:
        against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    against_QLUT_means = against_QLUT.mean()
    against_QLUT_sds = against_QLUT.std()
    against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

    try:
        against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    except:
        against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    against_QLDE_means = against_QLDE.mean()
    against_QLDE_sds = against_QLDE.std()
    against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

    try:
        against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    except:
        against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    against_QLVE_e_means = against_QLVE_e.mean()
    against_QLVE_e_sds = against_QLVE_e.std()
    against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

    try:
        against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    except:
        against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    against_QLVE_k_means = against_QLVE_k.mean()
    against_QLVE_k_sds = against_QLVE_k.std()
    against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)


    try:
        against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    except:
        against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
    against_QLVM_means = against_QLVM.mean()
    against_QLVM_sds = against_QLVM.std()
    against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)

    fig = plt.figure(figsize=(3, 3), dpi=80)
    plt.rcParams.update({'font.size':20})
    ax = fig.add_axes([0,0,1,1])
    #labels = ['vs_S', 'vs_UT', 'vs_DE', 'vs_'+r'$VE_e$', 'vs_'+r'$VE_k$'] 
    labels = ['S', 'UT', 'DE', 'VE'+r'$_e$', 'VE'+r'$_k$'] 
    means = [against_QLS_means,against_QLUT_means,against_QLDE_means,against_QLVE_e_means,against_QLVE_k_means]
    cis = [against_QLS_ci,against_QLUT_ci,against_QLDE_ci,against_QLVE_e_ci,against_QLVE_k_ci]
    colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple']  
    #add QLVM
    labels.append('VE'+r'$_m$')
    means.append(against_QLVM_means)
    cis.append(against_QLVM_ci)
    colors.append('pink')
    ax.bar(labels, means, yerr=cis, color=colors, width = 0.8) #capsize=7, 
    #plt.xticks(rotation=45)
    ax.set_ylabel(r'Cumulative $R_{intr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
    #(f'R^{extr}_{i=QLUT}'# for {player1_title}')
    long_title = title_mapping[player1_title].replace('Ethics','').replace('_','-')
    ax.set_title(long_title +' vs other \n') #'Cumulative Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
    ax.set_xlabel('Opponent type')
    if game_title=='IPD':
        if player1_title == 'QLUT':
            plt.gca().set_ylim([0, 60000])
        elif player1_title == 'QLDE':
            plt.gca().set_ylim([-50000, 0]) #[-1300, 0]
        elif player1_title == 'QLVE_e':
            plt.gca().set_ylim([0, 10000])
        elif player1_title == 'QLVE_k':
            plt.gca().set_ylim([0, 50000])
        elif player1_title == 'QLVM':
            plt.gca().set_ylim([0, 10000])
    elif game_title=='VOLUNTEER':
        if player1_title == 'QLUT':
            plt.gca().set_ylim([0, 80000])
        elif player1_title == 'QLDE':
            plt.gca().set_ylim([-50000, 0]) #[-1300, 0]
        elif player1_title == 'QLVE_e':
            plt.gca().set_ylim([0, 10000])
        elif player1_title == 'QLVE_k':
            plt.gca().set_ylim([0, 50000])
        elif player1_title == 'QLVM':
            plt.gca().set_ylim([0, 10000])
    elif game_title=='STAGHUNT': 
        if player1_title == 'QLUT':
            plt.gca().set_ylim([0, 100000])
        elif player1_title == 'QLDE':
            plt.gca().set_ylim([-50000, 0]) #[-1300, 0]
        elif player1_title == 'QLVE_e':
            plt.gca().set_ylim([0, 10000])
        elif player1_title == 'QLVE_k':
            plt.gca().set_ylim([0, 50000])
        elif player1_title == 'QLVM':
            plt.gca().set_ylim([0, 10000])
    plt.savefig(f'results/outcome_plots/reward/bar_cumulative_intrinsic_reward_{player1_title}.pdf', bbox_inches='tight')
    
    ##################################
    #### cumulative - game reward for player1_tytle vs others  ####
    ##################################
    against_QLS = pd.read_csv(f'results/{player1_title}_QLS/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
    against_QLS_means = against_QLS.mean(axis=1)
    against_QLS_sds = against_QLS.std(axis=1)
    against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

    try:
        against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
    except:
        against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0)
    against_QLUT_means = against_QLUT.mean(axis=1)
    against_QLUT_sds = against_QLUT.std(axis=1)
    against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

    try:
        against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
    except:
        against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0)
    against_QLDE_means = against_QLDE.mean(axis=1)
    against_QLDE_sds = against_QLDE.std(axis=1)
    against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

    try:
        against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
    except:
        against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0)
    against_QLVE_e_means = against_QLVE_e.mean(axis=1)
    against_QLVE_e_sds = against_QLVE_e.std(axis=1)
    against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

    try:
        against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
    except:
        against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0)
    against_QLVE_k_means = against_QLVE_k.mean(axis=1)
    against_QLVE_k_sds = against_QLVE_k.std(axis=1)
    against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)


    try:
        against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
    except:
        against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0)
    against_QLVM_means = against_QLVM.mean(axis=1)
    against_QLVM_sds = against_QLVM.std(axis=1)
    against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)

    plt.figure(dpi=80) #figsize=(10, 6)
    plt.plot(against_QLS.index[:], against_QLS_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLS', color='red')
    plt.fill_between(against_QLS.index[:], against_QLS_means-against_QLS_ci, against_QLS_means+against_QLS_ci, facecolor='#ff9999', alpha=0.5)
    plt.plot(against_QLUT.index[:], against_QLUT_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLUT', color='#556b2f')
    plt.fill_between(against_QLUT.index[:], against_QLUT_means-against_QLUT_ci, against_QLUT_means+against_QLUT_ci, facecolor='#ccff99', alpha=0.5)
    plt.plot(against_QLDE.index[:], against_QLDE_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLDE', color='#00cccc')
    plt.fill_between(against_QLDE.index[:], against_QLDE_means-against_QLDE_ci, against_QLDE_means+against_QLDE_ci, facecolor='#99ffff', alpha=0.5)
    plt.plot(against_QLVE_e.index[:], against_QLVE_e_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLVE_e', color='orange')
    plt.fill_between(against_QLVE_e.index[:], against_QLVE_e_means-against_QLVE_e_ci, against_QLVE_e_means+against_QLVE_e_ci, facecolor='#ffcc99', alpha=0.5)
    plt.plot(against_QLVE_k.index[:], against_QLVE_k_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLVE_k', color='purple')
    plt.fill_between(against_QLVE_k.index[:], against_QLVE_k_means-against_QLVE_k_ci, against_QLVE_k_means+against_QLVE_k_ci, facecolor='#CBC3E3', alpha=0.5)
    plt.plot(against_QLVM.index[:], against_QLVM_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_QLVM', color='pink')
    plt.fill_between(against_QLVM.index[:], against_QLVM_means-against_QLVM_ci, against_QLVM_means+against_QLVM_ci, facecolor='pink', alpha=0.5)
    
    plt.title(long_title +' vs other \n') #r'Cumulative Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
    if game_title=='IPD':
        if player1_title == 'QLUT':
            plt.gca().set_ylim([0, 60000])
        elif player1_title == 'QLDE':
            plt.gca().set_ylim([-50000, 0]) 
        elif player1_title == 'QLVE_e':
            plt.gca().set_ylim([0, 10000])
        elif player1_title == 'QLVE_k':
            plt.gca().set_ylim([0, 50000])
        elif player1_title == 'QLVM':
            plt.gca().set_ylim([0, 10000])
    elif game_title=='VOLUNTEER':
        if player1_title == 'QLUT':
            plt.gca().set_ylim([0, 80000])
        elif player1_title == 'QLDE':
            plt.gca().set_ylim([-50000, 0]) 
        elif player1_title == 'QLVE_e':
            plt.gca().set_ylim([0, 10000])
        elif player1_title == 'QLVE_k':
            plt.gca().set_ylim([0, 50000])
        elif player1_title == 'QLVM':
            plt.gca().set_ylim([0, 10000])
    elif game_title=='STAGHUNT': 
        if player1_title == 'QLUT':
            plt.gca().set_ylim([0, 100000])
        elif player1_title == 'QLDE':
            plt.gca().set_ylim([-50000, 0]) 
        elif player1_title == 'QLVE_e':
            plt.gca().set_ylim([0, 10000])
        elif player1_title == 'QLVE_k':
            plt.gca().set_ylim([0, 50000])
        elif player1_title == 'QLVM':
            plt.gca().set_ylim([0, 10000])
    plt.ylabel(r'Cumulative $R_{intr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
    plt.xlabel('Iteration')
    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    if not os.path.isdir('results/outcome_plots/reward'):
        os.makedirs('results/outcome_plots/reward')
    plt.savefig(f'results/outcome_plots/reward/cumulative_intrinsic_reward_{player1_title}.pdf', bbox_inches='tight')


    ##################################
    #### non-cumulative - game reward for player1_tytle vs others  ####
    ##################################
    figsize=(5, 4)

    against_QLS = pd.read_csv(f'results/{player1_title}_QLS/player1/df_reward_intrinsic.csv', index_col=0)
    against_QLS_means = against_QLS.mean(axis=1)
    against_QLS_sds = against_QLS.std(axis=1)
    against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

    try:
        against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/player1/df_reward_intrinsic.csv', index_col=0)
    except:
        against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/player2/df_reward_intrinsic.csv', index_col=0)
    against_QLUT_means = against_QLUT.mean(axis=1)
    against_QLUT_sds = against_QLUT.std(axis=1)
    against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

    try:
        against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/player1/df_reward_intrinsic.csv', index_col=0)
    except:
        against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/player2/df_reward_intrinsic.csv', index_col=0)
    against_QLDE_means = against_QLDE.mean(axis=1)
    against_QLDE_sds = against_QLDE.std(axis=1)
    against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

    try:
        against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/player1/df_reward_intrinsic.csv', index_col=0)
    except:
        against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/player2/df_reward_intrinsic.csv', index_col=0)
    against_QLVE_e_means = against_QLVE_e.mean(axis=1)
    against_QLVE_e_sds = against_QLVE_e.std(axis=1)
    against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

    try:
        against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/player1/df_reward_intrinsic.csv', index_col=0)
    except:
        against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/player2/df_reward_intrinsic.csv', index_col=0)
    against_QLVE_k_means = against_QLVE_k.mean(axis=1)
    against_QLVE_k_sds = against_QLVE_k.std(axis=1)
    against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

    try:
        against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/player1/df_reward_intrinsic.csv', index_col=0)
    except:
        against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/player2/df_reward_intrinsic.csv', index_col=0)
    against_QLVM_means = against_QLVM.mean(axis=1)
    against_QLVM_sds = against_QLVM.std(axis=1)
    against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)


    plt.figure(dpi=80, figsize=figsize)
    plt.rcParams.update({'font.size':20})
    plt.plot(against_QLS.index[:], against_QLS_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLS', color='red')
    plt.fill_between(against_QLS.index[:], against_QLS_means-against_QLS_ci, against_QLS_means+against_QLS_ci, facecolor='#ff9999', alpha=0.5)
    plt.plot(against_QLUT.index[:], against_QLUT_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLUT', color='#556b2f')
    plt.fill_between(against_QLUT.index[:], against_QLUT_means-against_QLUT_ci, against_QLUT_means+against_QLUT_ci, facecolor='#ccff99', alpha=0.5)
    plt.plot(against_QLDE.index[:], against_QLDE_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLDE', color='#00cccc')
    plt.fill_between(against_QLDE.index[:], against_QLDE_means-against_QLDE_ci, against_QLDE_means+against_QLDE_ci, facecolor='#99ffff', alpha=0.5)
    plt.plot(against_QLVE_e.index[:], against_QLVE_e_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLVE_e', color='orange')
    plt.fill_between(against_QLVE_e.index[:], against_QLVE_e_means-against_QLVE_e_ci, against_QLVE_e_means+against_QLVE_e_ci, facecolor='#ffcc99', alpha=0.5)
    plt.plot(against_QLVE_k.index[:], against_QLVE_k_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLVE_k', color='purple')
    plt.fill_between(against_QLVE_k.index[:], against_QLVE_k_means-against_QLVE_k_ci, against_QLVE_k_means+against_QLVE_k_ci, facecolor='#CBC3E3', alpha=0.5)
    plt.plot(against_QLVM.index[:], against_QLVM_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_QLVM', color='palevioletred')
    plt.fill_between(against_QLVM.index[:], against_QLVM_means-against_QLVM_ci, against_QLVM_means+against_QLVM_ci, facecolor='pink', alpha=0.5)
        
    plt.title(long_title +' vs other \n') #r'Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 

    if game_title=='IPD':
        if player1_title == 'QLUT':
            plt.gca().set_ylim([4, 6])
        elif player1_title == 'QLDE':
            plt.gca().set_ylim([-5, 0]) 
        elif player1_title == 'QLVE_e':
            plt.gca().set_ylim([0, 1])
        elif player1_title == 'QLVE_k':
            plt.gca().set_ylim([0, 5])
    elif game_title=='VOLUNTEER':
        if player1_title == 'QLUT':
            plt.gca().set_ylim([2, 8])
        elif player1_title == 'QLDE':
            plt.gca().set_ylim([-5, 0]) 
        elif player1_title == 'QLVE_e':
            plt.gca().set_ylim([0, 1])
        elif player1_title == 'QLVE_k':
            plt.gca().set_ylim([0, 5])
    elif game_title=='STAGHUNT': 
        if player1_title == 'QLUT':
            plt.gca().set_ylim([4, 10])
        elif player1_title == 'QLDE':
            plt.gca().set_ylim([-5, 0]) 
        elif player1_title == 'QLVE_e':
            plt.gca().set_ylim([0, 1])
        elif player1_title == 'QLVE_k':
            plt.gca().set_ylim([0, 5])

    plt.ylabel(r'$R_{intr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
    plt.xlabel('Iteration')
    #plt.xticks(rotation=45)
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)
    plt.savefig(f'results/outcome_plots/reward/intrinsic_reward_{player1_title}.pdf', bbox_inches='tight')

def plot_relative_action_pairs(player1_title, n_runs):
    '''visualise % mutual cooperation
    - consider the whole run'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    figsize=(5, 4)

    action_pairs_against_QLS = pd.read_csv(f'results/{player1_title}_QLS/action_pairs.csv', index_col=0)
    try:
        action_pairs_against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/action_pairs.csv', index_col=0)
        order_QLUT = ''
    except:
        action_pairs_against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/action_pairs.csv', index_col=0)
        order_QLUT = 'QLUT_first'
    try:
        action_pairs_against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/action_pairs.csv', index_col=0)
        order_QLDE = ''
    except:
        action_pairs_against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/action_pairs.csv', index_col=0)
        order_QLDE = 'QLDE_first'
    try:
        action_pairs_against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/action_pairs.csv', index_col=0)
        order_QLVE_e = ''
    except:
        action_pairs_against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/action_pairs.csv', index_col=0)
        order_QLVE_e = 'QLVE_e_first'
    try:
        action_pairs_against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/action_pairs.csv', index_col=0)
        order_QLVE_k = ''
    except:
        action_pairs_against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/action_pairs.csv', index_col=0)
        order_QLVE_k = 'QLVE_k_first'
    if player1_title == 'QLVM':
        try:
            action_pairs_against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/action_pairs.csv', index_col=0)
            order_QLVM = ''
        except:
            action_pairs_against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/action_pairs.csv', index_col=0)
            order_QLVM = 'QLVM_first'


    ################################
    #### plot mutual cooperation ####
    ################################
    action_pairs_against_QLS['%_CC'] = (action_pairs_against_QLS[action_pairs_against_QLS[:]=='C, C'].count(axis='columns') / n_runs)*100
    action_pairs_against_QLUT['%_CC'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='C, C'].count(axis='columns') / n_runs)*100
    action_pairs_against_QLDE['%_CC'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='C, C'].count(axis='columns') / n_runs)*100
    action_pairs_against_QLVE_e['%_CC'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='C, C'].count(axis='columns') / n_runs)*100
    action_pairs_against_QLVE_k['%_CC'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='C, C'].count(axis='columns') / n_runs)*100
    if player1_title == 'QLVM':
        action_pairs_against_QLVM['%_CC'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='C, C'].count(axis='columns') / n_runs)*100


    #plot results 
    plt.figure(dpi=80, figsize=figsize)
    plt.rcParams.update({'font.size':20})
    colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple']
    plt.plot(action_pairs_against_QLS.index[:], action_pairs_against_QLS['%_CC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLS', color=colors[0])
    plt.plot(action_pairs_against_QLUT.index[:], action_pairs_against_QLUT['%_CC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLUT', color=colors[1])
    plt.plot(action_pairs_against_QLDE.index[:], action_pairs_against_QLDE['%_CC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLDE', color=colors[2])
    plt.plot(action_pairs_against_QLVE_e.index[:], action_pairs_against_QLVE_e['%_CC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_e', color=colors[3])
    plt.plot(action_pairs_against_QLVE_k.index[:], action_pairs_against_QLVE_k['%_CC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_k', color=colors[4])
    if player1_title == 'QLVM':
        plt.plot(action_pairs_against_QLVM.index[:], action_pairs_against_QLVM['%_CC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVM', color='pink')
    plt.title('Mutual C - '+ player1_title.replace('QL','') +' vs other \n') # among the two agents (% over ' +str(n_runs)+' runs), '+'\n'+ 
    plt.gca().set_ylim([0, 100])
    plt.ylabel('% C,C (over 100 runs)')
    plt.xlabel('Iteration')
    #plt.xticks(rotation=45)
    if player1_title != 'QLVM':
        leg = plt.legend(fontsize=14, labels=['vs S', 'vs UT', 'vs DE', r'vs VE$_e$', r'vs VE$_k$']) # get the legend object
    else: 
        leg = plt.legend(fontsize=14, labels=['vs S', 'vs UT', 'vs DE', r'vs VE$_e$', r'vs VE$_k$', 'vs VM']) # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(8)
    if not os.path.isdir('results/outcome_plots/cooperation'):
        os.makedirs('results/outcome_plots/cooperation')
    plt.savefig(f'results/outcome_plots/cooperation/mutual_cooperation_{player1_title}.png', bbox_inches='tight')


    ################################
    #### plot mutual defection ####
    ################################
    action_pairs_against_QLS['%_DD'] = (action_pairs_against_QLS[action_pairs_against_QLS[:]=='D, D'].count(axis='columns')/ n_runs)*100
    action_pairs_against_QLUT['%_DD'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='D, D'].count(axis='columns')/ n_runs)*100
    action_pairs_against_QLDE['%_DD'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='D, D'].count(axis='columns')/ n_runs)*100
    action_pairs_against_QLVE_e['%_DD'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='D, D'].count(axis='columns')/ n_runs)*100
    action_pairs_against_QLVE_k['%_DD'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='D, D'].count(axis='columns')/ n_runs)*100
    if player1_title == 'QLVM':
        action_pairs_against_QLVM['%_DD'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='D, D'].count(axis='columns') / n_runs)*100

    #plot results 
    plt.figure(dpi=80, figsize=figsize) 
    plt.rcParams.update({'font.size':20})
    colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple']
    plt.plot(action_pairs_against_QLS.index[:], action_pairs_against_QLS['%_DD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLS', color=colors[0])
    plt.plot(action_pairs_against_QLUT.index[:], action_pairs_against_QLUT['%_DD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLUT', color=colors[1])
    plt.plot(action_pairs_against_QLDE.index[:], action_pairs_against_QLDE['%_DD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLDE', color=colors[2])
    plt.plot(action_pairs_against_QLVE_e.index[:], action_pairs_against_QLVE_e['%_DD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_e', color=colors[3])
    plt.plot(action_pairs_against_QLVE_k.index[:], action_pairs_against_QLVE_k['%_DD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_k',color=colors[4])
    if player1_title == 'QLVM':
        plt.plot(action_pairs_against_QLVM.index[:], action_pairs_against_QLVM['%_DD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVM', color=colors[4])
    plt.title('Mutual D - '+ player1_title.replace('QL','') +' vs other \n') #'Mutual Defection among the two agents (% over ' +str(n_runs)+' runs), '+'\n'+
    plt.gca().set_ylim([0, 100])
    plt.ylabel('% D,D (over 100 runs)')
    plt.xlabel('Iteration')
    #plt.xticks(rotation=45)
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)
    plt.savefig(f'results/outcome_plots/cooperation/mutual_defection_{player1_title}.png', bbox_inches='tight')


    ################################
    #### plot player1 eploits player2 ####
    ################################
    action_pairs_against_QLS['%_DC'] = (action_pairs_against_QLS[action_pairs_against_QLS[:]=='D, C'].count(axis='columns')/ n_runs)*100
    if order_QLUT=='QLUT_first':
        action_pairs_against_QLUT['%_DC'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='C, D'].count(axis='columns')/ n_runs)*100
    else: 
        action_pairs_against_QLUT['%_DC'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='D, C'].count(axis='columns')/ n_runs)*100
    if order_QLDE=='QLDE_first':
        action_pairs_against_QLDE['%_DC'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='C, D'].count(axis='columns')/ n_runs)*100
    else: 
        action_pairs_against_QLDE['%_DC'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='D, C'].count(axis='columns')/ n_runs)*100
    if order_QLVE_e=='QLVE_e_first':
        action_pairs_against_QLVE_e['%_DC'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='C, D'].count(axis='columns')/ n_runs)*100
    else: 
        action_pairs_against_QLVE_e['%_DC'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='D, C'].count(axis='columns')/ n_runs)*100
    if order_QLVE_k=='QLVE_k_first':
        action_pairs_against_QLVE_k['%_DC'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='C, D'].count(axis='columns')/ n_runs)*100
    else:
        action_pairs_against_QLVE_k['%_DC'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='D, C'].count(axis='columns')/ n_runs)*100
    if player1_title == 'QLVM':
        if order_QLVM=='QLVM_first':
            action_pairs_against_QLVM['%_DC'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='C, D'].count(axis='columns')/ n_runs)*100
        else:
            action_pairs_against_QLVM['%_DC'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='D, C'].count(axis='columns')/ n_runs)*100
    

    #plot results 
    plt.figure(dpi=80, figsize=figsize) 
    plt.rcParams.update({'font.size':20})
    colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple']
    plt.plot(action_pairs_against_QLS.index[:], action_pairs_against_QLS['%_DC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLS', color=colors[0])
    plt.plot(action_pairs_against_QLUT.index[:], action_pairs_against_QLUT['%_DC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLUT', color=colors[1])
    plt.plot(action_pairs_against_QLDE.index[:], action_pairs_against_QLDE['%_DC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLDE', color=colors[2])
    plt.plot(action_pairs_against_QLVE_e.index[:], action_pairs_against_QLVE_e['%_DC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_e', color=colors[3])
    plt.plot(action_pairs_against_QLVE_k.index[:], action_pairs_against_QLVE_k['%_DC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_k', color=colors[4])
    if player1_title == 'QLVM':
        plt.plot(action_pairs_against_QLVM.index[:], action_pairs_against_QLVM['%_DC'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVM', color=colors[4])
    plt.title(player1_title.replace('QL','') +' exploits other \n') #'Exploitation among the two agents (% over ' +str(n_runs)+' runs), '+'\n'+ 
    plt.gca().set_ylim([0, 100])
    plt.ylabel('% D,C (over 100 runs)') #\n player i exploits other
    plt.xlabel('Iteration')
    #plt.xticks(rotation=45)
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)
    plt.savefig(f'results/outcome_plots/cooperation/exploitation_{player1_title}.png', bbox_inches='tight')


    ################################
    #### plot player1 gets exploited by player2 ####
    ################################
    action_pairs_against_QLS['%_CD'] = (action_pairs_against_QLS[action_pairs_against_QLS[:]=='C, D'].count(axis='columns')/ n_runs)*100
    if order_QLUT=='QLUT_first':
        action_pairs_against_QLUT['%_CD'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='D, C'].count(axis='columns')/ n_runs)*100
    else: 
        action_pairs_against_QLUT['%_CD'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='C, D'].count(axis='columns')/ n_runs)*100
    if order_QLDE=='QLDE_first':
        action_pairs_against_QLDE['%_CD'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='D, C'].count(axis='columns')/ n_runs)*100
    else: 
        action_pairs_against_QLDE['%_CD'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='C, D'].count(axis='columns')/ n_runs)*100
    if order_QLVE_e=='QLVE_e_first':
        action_pairs_against_QLVE_e['%_CD'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='D, C'].count(axis='columns')/ n_runs)*100
    else: 
        action_pairs_against_QLVE_e['%_CD'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='C, D'].count(axis='columns')/ n_runs)*100
    if order_QLVE_k=='QLVE_k_first':
        action_pairs_against_QLVE_k['%_CD'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='D, C'].count(axis='columns')/ n_runs)*100
    else:
        action_pairs_against_QLVE_k['%_CD'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='C, D'].count(axis='columns')/ n_runs)*100
    if player1_title == 'QLVM':
        if order_QLVM=='QLVM_first':
            action_pairs_against_QLVM['%_CD'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='D, C'].count(axis='columns')/ n_runs)*100
        else:
            action_pairs_against_QLVM['%_CD'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='C, D'].count(axis='columns')/ n_runs)*100
    
    #plot results 
    plt.figure(dpi=80, figsize=figsize)
    plt.rcParams.update({'font.size':20})
    colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple']
    plt.plot(action_pairs_against_QLS.index[:], action_pairs_against_QLS['%_CD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLS', color=colors[0])
    plt.plot(action_pairs_against_QLUT.index[:], action_pairs_against_QLUT['%_CD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLUT', color=colors[1])
    plt.plot(action_pairs_against_QLDE.index[:], action_pairs_against_QLDE['%_CD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLDE', color=colors[2])
    plt.plot(action_pairs_against_QLVE_e.index[:], action_pairs_against_QLVE_e['%_CD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_e', color=colors[3])
    plt.plot(action_pairs_against_QLVE_k.index[:], action_pairs_against_QLVE_k['%_CD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_k', color=colors[4])
    if player1_title == 'QLVM':
        plt.plot(action_pairs_against_QLVM.index[:], action_pairs_against_QLVM['%_CD'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVM', color=colors[4])
    plt.title(player1_title.replace('QL','') +' gets exploited \n') #Exploitation among the two agents (% over ' +str(n_runs)+' runs), '+'\n'+ 
    plt.gca().set_ylim([0, 100])
    plt.ylabel('% C,D (over 100 runs)') #\n player i gets exploited by other
    plt.xlabel('Iteration')
    #plt.xticks(rotation=45)
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)
    plt.savefig(f'results/outcome_plots/cooperation/gets_exploited_{player1_title}.png', bbox_inches='tight')

def plot_relative_outcomes(type, player1_title, n_runs, game_title):
    '''plot different types of social outcoes - collective / gini / min (game) reward for different pairs'''    
    ##################################
    #### cumulative - {type} game reward for player1_tytle vs others  ####
    ##################################
    against_QLS = pd.read_csv(f'results/{player1_title}_QLS/df_cumulative_reward_{type}.csv', index_col=0)
    against_QLS_means = against_QLS.mean(axis=1)
    against_QLS_sds = against_QLS.std(axis=1)
    against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

    try:
        against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/df_cumulative_reward_{type}.csv', index_col=0)
    except:
        against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0)
    against_QLUT_means = against_QLUT.mean(axis=1)
    against_QLUT_sds = against_QLUT.std(axis=1)
    against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

    try:
        against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/df_cumulative_reward_{type}.csv', index_col=0)
    except:
        against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0)
    against_QLDE_means = against_QLDE.mean(axis=1)
    against_QLDE_sds = against_QLDE.std(axis=1)
    against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

    try:
        against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/df_cumulative_reward_{type}.csv', index_col=0)
    except:
        against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0)
    against_QLVE_e_means = against_QLVE_e.mean(axis=1)
    against_QLVE_e_sds = against_QLVE_e.std(axis=1)
    against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

    try:
        against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/df_cumulative_reward_{type}.csv', index_col=0)
    except:
        against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0)
    against_QLVE_k_means = against_QLVE_k.mean(axis=1)
    against_QLVE_k_sds = against_QLVE_k.std(axis=1)
    against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

    try:
        against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/df_cumulative_reward_{type}.csv', index_col=0)
    except:
        against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0)
    against_QLVM_means = against_QLVM.mean(axis=1)
    against_QLVM_sds = against_QLVM.std(axis=1)
    against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)

    plt.figure(dpi=80, figsize=(5,4)) 
    plt.rcParams.update({'font.size':20})
    plt.plot(against_QLS.index[:], against_QLS_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_S', color='red')
    plt.fill_between(against_QLS.index[:], against_QLS_means-against_QLS_ci, against_QLS_means+against_QLS_ci, facecolor='#ff9999', alpha=0.5)
    plt.plot(against_QLUT.index[:], against_QLUT_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_UT', color='#556b2f')
    plt.fill_between(against_QLUT.index[:], against_QLUT_means-against_QLUT_ci, against_QLUT_means+against_QLUT_ci, facecolor='#ccff99', alpha=0.5)
    plt.plot(against_QLDE.index[:], against_QLDE_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_DE', color='#00cccc')
    plt.fill_between(against_QLDE.index[:], against_QLDE_means-against_QLDE_ci, against_QLDE_means+against_QLDE_ci, facecolor='#99ffff', alpha=0.5)
    plt.plot(against_QLVE_e.index[:], against_QLVE_e_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_'+r'$VE_e$', color='orange')
    plt.fill_between(against_QLVE_e.index[:], against_QLVE_e_means-against_QLVE_e_ci, against_QLVE_e_means+against_QLVE_e_ci, facecolor='#ffcc99', alpha=0.5)
    plt.plot(against_QLVE_k.index[:], against_QLVE_k_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_'+r'$VE_k$', color='purple')
    plt.fill_between(against_QLVE_k.index[:], against_QLVE_k_means-against_QLVE_k_ci, against_QLVE_k_means+against_QLVE_k_ci, facecolor='#CBC3E3', alpha=0.5)
    plt.plot(against_QLVM.index[:], against_QLVM_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_'+r'$VE_m$', color='palevioletred')
    plt.fill_between(against_QLVM.index[:], against_QLVM_means-against_QLVM_ci, against_QLVM_means+against_QLVM_ci, facecolor='pink', alpha=0.5)
    long_title = title_mapping[player1_title].replace('Ethics','').replace('_','-')
    plt.title(long_title +' vs other \n') #r'Cumulative '+type+' Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
    if game_title=='IPD':
        if type=='collective':
            plt.gca().set_ylim([0, 60000])
        elif type=='gini':
            plt.gca().set_ylim([0, 10000])
        elif type=='min':
            plt.gca().set_ylim([0, 30000])
    elif game_title=='VOLUNTEER':
        if type=='collective': 
            plt.gca().set_ylim([0, 80000])
        elif type=='gini':
            plt.gca().set_ylim([0, 10000])
        elif type=='min':
            plt.gca().set_ylim([0, 40000])
    elif game_title=='STAGHUNT': 
        if type=='collective': 
            plt.gca().set_ylim([0, 100000])
        elif type=='gini':
            plt.gca().set_ylim([0, 10000])
        elif type=='min':
            plt.gca().set_ylim([0, 40000])  
    plt.ylabel(f'$G_{{{type}}}$') #Cumulative
    plt.xlabel('Iteration')
    #plt.xticks(rotation=45)
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)
    if not os.path.isdir('results/outcome_plots/group_outcomes'):
        os.makedirs('results/outcome_plots/group_outcomes')

    #plt.savefig(f'results/outcome_plots/group_outcomes/cumulative_{type}_reward_{player1_title}.png', bbox_inches='tight')

    

    

    ######################################
    #### cumulative bar plot - {type} game reward ####
    ######################################
    against_QLS = pd.read_csv(f'results/{player1_title}_QLS/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    against_QLS_means = against_QLS.mean()
    against_QLS_sds = against_QLS.std()
    against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

    try:
        against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    except:
        against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    against_QLUT_means = against_QLUT.mean()
    against_QLUT_sds = against_QLUT.std()
    against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

    try:
        against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    except:
        against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    against_QLDE_means = against_QLDE.mean()
    against_QLDE_sds = against_QLDE.std()
    against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

    try:
        against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    except:
        against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    against_QLVE_e_means = against_QLVE_e.mean()
    against_QLVE_e_sds = against_QLVE_e.std()
    against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

    try:
        against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    except:
        against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    against_QLVE_k_means = against_QLVE_k.mean()
    against_QLVE_k_sds = against_QLVE_k.std()
    against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

    try:
        against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    except:
        against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
    against_QLVM_means = against_QLVM.mean()
    against_QLVM_sds = against_QLVM.std()
    against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)


    #import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3, 3), dpi=80)
    plt.rcParams.update({'font.size':20})
    ax = fig.add_axes([0,0,1,1])
    #labels = ['vs_S', 'vs_UT', 'vs_DE', 'vs_'+r'$VE_e$', 'vs_'+r'$VE_k$']
    labels = ['S', 'UT', 'DE', 'VE'+r'$_e$', 'VE'+r'$_k$', 'VE'+r'$_m$']
    means = [against_QLS_means,against_QLUT_means,against_QLDE_means,against_QLVE_e_means,against_QLVE_k_means,against_QLVM_means]
    cis = [against_QLS_ci,against_QLUT_ci,against_QLDE_ci,against_QLVE_e_ci,against_QLVE_k_ci,against_QLVM_ci]
    colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple', 'pink']
    ax.bar(labels, means, yerr=cis, capsize=7, color=colors)
    ax.set_title(long_title +' vs other \n') #r'Cumulative '+type+' Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
    if game_title=='IPD':
        if type=='collective':
            plt.gca().set_ylim([0, 60000])
        elif type=='gini':
            plt.gca().set_ylim([0, 10000])
        elif type=='min':
            plt.gca().set_ylim([0, 30000])
    elif game_title=='VOLUNTEER':
        if type=='collective': 
            plt.gca().set_ylim([0, 80000])
        elif type=='gini':
            plt.gca().set_ylim([0, 10000])
        elif type=='min':
            plt.gca().set_ylim([0, 40000])
    elif game_title=='STAGHUNT': 
        if type=='collective': 
            plt.gca().set_ylim([0, 100000])
        elif type=='gini':
            plt.gca().set_ylim([0, 10000])
        elif type=='min':
            plt.gca().set_ylim([0, 40000])  
    ax.set_ylabel(f'$G_{{{type}}}$') #Cumulative
    ax.set_xlabel('Opponent type')
    #plt.xticks(rotation=45)
    plt.savefig(f'results/outcome_plots/group_outcomes/bar_cumulative_{type}_reward_{player1_title}.pdf', bbox_inches='tight')



    ######################################
    #### non-cumulative - {type} game reward ####
    ######################################
    #my_df = pd.read_csv(f'{destination_folder}/df_reward_collective.csv', index_col=0)
    #means = my_df.mean(axis=1)
    #sds = my_df.std(axis=1)

    #plt.figure(dpi=80) #figsize=(10, 6)
    #plt.plot(my_df.index[:], means[:], lw=0.5, label=f'both players', color='purple')
    #plt.fill_between(my_df.index[:], means-sds, means+sds, facecolor='#bf92e4', alpha=0.7)
    #plt.title(r'Collective Reward (Mean over ' +str(n_runs)+r' runs $\pm$ SD), '+'\n'+player1_title+' vs '+player2_title)
    #plt.ylabel('Collective reward')
    #plt.xlabel('Iteration')
    #plt.legend() #loc='upper left'
    #plt.savefig(f'{destination_folder}/plots/Collective_reward.png', bbox_inches='tight') 
    against_QLS = pd.read_csv(f'results/{player1_title}_QLS/df_reward_{type}.csv', index_col=0)
    against_QLS_means = against_QLS.mean(axis=1)
    against_QLS_sds = against_QLS.std(axis=1)
    against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

    try:
        against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/df_reward_{type}.csv', index_col=0)
    except:
        against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/df_reward_{type}.csv', index_col=0)
    against_QLUT_means = against_QLUT.mean(axis=1)
    against_QLUT_sds = against_QLUT.std(axis=1)
    against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

    try:
        against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/df_reward_{type}.csv', index_col=0)
    except:
        against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/df_reward_{type}.csv', index_col=0)
    against_QLDE_means = against_QLDE.mean(axis=1)
    against_QLDE_sds = against_QLDE.std(axis=1)
    against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

    try:
        against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/df_reward_{type}.csv', index_col=0)
    except:
        against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/df_reward_{type}.csv', index_col=0)
    against_QLVE_e_means = against_QLVE_e.mean(axis=1)
    against_QLVE_e_sds = against_QLVE_e.std(axis=1)
    against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

    try:
        against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/df_reward_{type}.csv', index_col=0)
    except:
        against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/df_reward_{type}.csv', index_col=0)
    against_QLVE_k_means = against_QLVE_k.mean(axis=1)
    against_QLVE_k_sds = against_QLVE_k.std(axis=1)
    against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

    try:
        against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/df_reward_{type}.csv', index_col=0)
    except:
        against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/df_reward_{type}.csv', index_col=0)
    against_QLVM_means = against_QLVM.mean(axis=1)
    against_QLVM_sds = against_QLVM.std(axis=1)
    against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)

    plt.figure(dpi=80, figsize=(5,4))
    plt.rcParams.update({'font.size':20})
    plt.plot(against_QLS.index[:], against_QLS_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_S', color='red')
    plt.fill_between(against_QLS.index[:], against_QLS_means-against_QLS_ci, against_QLS_means+against_QLS_ci, facecolor='#ff9999', alpha=0.5)
    plt.plot(against_QLUT.index[:], against_QLUT_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_UT', color='#556b2f')
    plt.fill_between(against_QLUT.index[:], against_QLUT_means-against_QLUT_ci, against_QLUT_means+against_QLUT_ci, facecolor='#ccff99', alpha=0.5)
    plt.plot(against_QLDE.index[:], against_QLDE_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_DE', color='#00cccc')
    plt.fill_between(against_QLDE.index[:], against_QLDE_means-against_QLDE_ci, against_QLDE_means+against_QLDE_ci, facecolor='#99ffff', alpha=0.5)
    plt.plot(against_QLVE_e.index[:], against_QLVE_e_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_'+'VE'+r'$_e$', color='orange')
    plt.fill_between(against_QLVE_e.index[:], against_QLVE_e_means-against_QLVE_e_ci, against_QLVE_e_means+against_QLVE_e_ci, facecolor='#ffcc99', alpha=0.5)
    plt.plot(against_QLVE_k.index[:], against_QLVE_k_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_'+'VE'+r'$_k$', color='purple')
    plt.fill_between(against_QLVE_k.index[:], against_QLVE_k_means-against_QLVE_k_ci, against_QLVE_k_means+against_QLVE_k_ci, facecolor='#CBC3E3', alpha=0.5)
    plt.plot(against_QLVM.index[:], against_QLVM_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_'+'VE'+r'$_m$', color='palevioletred')
    plt.fill_between(against_QLVM.index[:], against_QLVM_means-against_QLVM_ci, against_QLVM_means+against_QLVM_ci, facecolor='pink', alpha=0.5)

    plt.title(long_title +' vs other \n') #type+r' Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
    if type=='gini':
        plt.gca().set_ylim([0, 1])

    if game_title=='IPD':
        if type=='collective':
            plt.gca().set_ylim([4, 6])
        elif type=='min':
            plt.gca().set_ylim([1, 3])
    elif game_title=='VOLUNTEER':
        if type=='collective': 
            plt.gca().set_ylim([2, 8])
        elif type=='min':
            plt.gca().set_ylim([1, 4])
    elif game_title=='STAGHUNT': 
        if type=='collective': 
            plt.gca().set_ylim([4, 10])
        elif type=='min':
            plt.gca().set_ylim([1, 5])
    plt.ylabel(f'$G_{{{type}}}$')
    plt.xlabel('Iteration')
    #leg = plt.legend(fontsize=14, labels=['vs QLS', 'vs QLUT', 'vs QLDE', r'vs QLVE$_e$', r'vs QLVE$_k$']) # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(8)
    #plt.savefig(f'results/outcome_plots/group_outcomes/{type}_reward_{player1_title}.png', bbox_inches='tight')


def create_legend():
    player1_title = 'QLUT'
    action_pairs_against_QLS = pd.read_csv(f'results/{player1_title}_QLS/action_pairs.csv', index_col=0)
    try:
        action_pairs_against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/action_pairs.csv', index_col=0)
        order_QLUT = ''
    except:
        action_pairs_against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/action_pairs.csv', index_col=0)
        order_QLUT = 'QLUT_first'
    try:
        action_pairs_against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/action_pairs.csv', index_col=0)
        order_QLDE = ''
    except:
        action_pairs_against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/action_pairs.csv', index_col=0)
        order_QLDE = 'QLDE_first'
    try:
        action_pairs_against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/action_pairs.csv', index_col=0)
        order_QLVE_e = ''
    except:
        action_pairs_against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/action_pairs.csv', index_col=0)
        order_QLVE_e = 'QLVE_e_first'
    try:
        action_pairs_against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/action_pairs.csv', index_col=0)
        order_QLVE_k = ''
    except:
        action_pairs_against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/action_pairs.csv', index_col=0)
        order_QLVE_k = 'QLVE_k_first'
    try:
        action_pairs_against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/action_pairs.csv', index_col=0)
        order_QLVEM = ''
    except:
        action_pairs_against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/action_pairs.csv', index_col=0)
        order_QLVM = 'QLVM_first'


    ################################
    #### plot mutual cooperation ####
    ################################
    action_pairs_against_QLS['%_CC'] = action_pairs_against_QLS[action_pairs_against_QLS[:]=='C, C'].count(axis='columns')
    action_pairs_against_QLUT['%_CC'] = action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='C, C'].count(axis='columns')
    action_pairs_against_QLDE['%_CC'] = action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='C, C'].count(axis='columns')
    action_pairs_against_QLVE_e['%_CC'] = action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='C, C'].count(axis='columns')
    action_pairs_against_QLVE_k['%_CC'] = action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='C, C'].count(axis='columns')
    action_pairs_against_QLVM['%_CC'] = action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='C, C'].count(axis='columns')

    colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple', 'palevioletred']

    #plot results - these will be ignored and not stored, just to make the plot 
    plt.figure(dpi=80) #figsize=(10, 6), 
    plt.rcParams.update({'font.size':20})
    plt.plot(action_pairs_against_QLS.index[:], action_pairs_against_QLS['%_CC'], lw=0.5, alpha=0.5, label=f'{player1_title} vs. QLS', color=colors[0])

    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot", figsize=(2.5, 2.5))
    ax = fig.add_subplot(111)
    line1, = ax.plot(action_pairs_against_QLS.index[:], action_pairs_against_QLS['%_CC'], c=colors[0], lw=6)
    line2, = ax.plot(action_pairs_against_QLUT.index[:], action_pairs_against_QLUT['%_CC'], c=colors[1], lw=6)
    line3, = ax.plot(action_pairs_against_QLDE.index[:], action_pairs_against_QLDE['%_CC'], c=colors[2], lw=6)
    line4, = ax.plot(action_pairs_against_QLVE_e.index[:], action_pairs_against_QLVE_e['%_CC'], c=colors[3], lw=6)
    line5, = ax.plot(action_pairs_against_QLVE_k.index[:], action_pairs_against_QLVE_k['%_CC'], c=colors[4], lw=6)
    line6, = ax.plot(action_pairs_against_QLVM.index[:], action_pairs_against_QLVM['%_CC'], c=colors[5], lw=6)

    legendFig.legend([line1, line2, line3, line4, line5, line6], ['vs Selfish', 'vs UT', 'vs DE', r'vs VE$_e$', r'vs VE$_k$', r'vs VE$_m$'], loc='center')
    legendFig.savefig('results/outcome_plots/legend.pdf', bbox='tight', transparent=True)
#create_legend()


def plot_relative_cooperation(player1_title, n_runs): 
    actions_against_QLS = pd.read_csv(f'results/{player1_title}_QLS/player1/action.csv', index_col=0)
    #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
    actions_against_QLS['%_defect'] = actions_against_QLS[actions_against_QLS[:]==1].count(axis='columns')
    actions_against_QLS['%_cooperate'] = n_runs-actions_against_QLS['%_defect']
    actions_against_QLS['%_defect'] = (actions_against_QLS['%_defect']/n_runs)*100
    actions_against_QLS['%_cooperate'] = (actions_against_QLS['%_cooperate']/n_runs)*100

    try:
        actions_against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/player1/action.csv', index_col=0)
    except:
        actions_against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/player2/action.csv', index_col=0)
    actions_against_QLUT['%_defect'] = actions_against_QLUT[actions_against_QLUT[:]==1].count(axis='columns')
    actions_against_QLUT['%_cooperate'] = n_runs-actions_against_QLUT['%_defect']
    actions_against_QLUT['%_defect'] = (actions_against_QLUT['%_defect']/n_runs)*100
    actions_against_QLUT['%_cooperate'] = (actions_against_QLUT['%_cooperate']/n_runs)*100

    try:
        actions_against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/player1/action.csv', index_col=0)
    except:
        actions_against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/player2/action.csv', index_col=0)
    actions_against_QLDE['%_defect'] = actions_against_QLDE[actions_against_QLDE[:]==1].count(axis='columns')
    actions_against_QLDE['%_cooperate'] = n_runs-actions_against_QLDE['%_defect']
    actions_against_QLDE['%_defect'] = (actions_against_QLDE['%_defect']/n_runs)*100
    actions_against_QLDE['%_cooperate'] = (actions_against_QLDE['%_cooperate']/n_runs)*100

    try:
        actions_against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/player1/action.csv', index_col=0)
    except:
        actions_against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/player2/action.csv', index_col=0)
    actions_against_QLVE_e['%_defect'] = actions_against_QLVE_e[actions_against_QLVE_e[:]==1].count(axis='columns')
    actions_against_QLVE_e['%_cooperate'] = n_runs-actions_against_QLVE_e['%_defect']
    actions_against_QLVE_e['%_defect'] = (actions_against_QLVE_e['%_defect']/n_runs)*100
    actions_against_QLVE_e['%_cooperate'] = (actions_against_QLVE_e['%_cooperate']/n_runs)*100

    try:
        actions_against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/player1/action.csv', index_col=0)
    except:
        actions_against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/player2/action.csv', index_col=0)
    actions_against_QLVE_k['%_defect'] = actions_against_QLVE_k[actions_against_QLVE_k[:]==1].count(axis='columns')
    actions_against_QLVE_k['%_cooperate'] = n_runs-actions_against_QLVE_k['%_defect']
    actions_against_QLVE_k['%_defect'] = (actions_against_QLVE_k['%_defect']/n_runs)*100
    actions_against_QLVE_k['%_cooperate'] = (actions_against_QLVE_k['%_cooperate']/n_runs)*100

    #plot results 
    plt.figure(dpi=80) #figsize=(10, 6), 
    colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple']
    plt.plot(actions_against_QLS.index[:], actions_against_QLS['%_cooperate'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLS', color=colors[0])
    plt.plot(actions_against_QLUT.index[:], actions_against_QLUT['%_cooperate'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLUT', color=colors[1])
    plt.plot(actions_against_QLDE.index[:], actions_against_QLDE['%_cooperate'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLDE', color=colors[2])
    plt.plot(actions_against_QLVE_e.index[:], actions_against_QLVE_e['%_cooperate'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_e', color=colors[3])
    plt.plot(actions_against_QLVE_k.index[:], actions_against_QLVE_k['%_cooperate'], lw=0.1, alpha=0.5, label=f'{player1_title} vs. QLVE_k', color=colors[4])
    long_title = title_mapping[player1_title]
    plt.title(long_title +' vs other') #Probability of Cooperation,'+'\n'+ 
    plt.gca().set_ylim([0, 100])
    plt.ylabel('% '+r'$a_M=C$'+' (over ' +str(n_runs)+' runs)')
    plt.xlabel('Iteration')
    #leg = plt.legend() # get the legend object
    #for line in leg.get_lines(): # change the line width for the legend
    #    line.set_linewidth(4.0)
    if not os.path.isdir('results/outcome_plots/cooperation'):
        os.makedirs('results/outcome_plots/cooperation')
    plt.savefig(f'results/outcome_plots/cooperation/cooperation_{player1_title}.png', bbox_inches='tight')

def C_pairs_condition(v):
        if v == 'C, C':
            color = "#28641E"
        elif v =='C, D':
            color = "#B0DC82"
        elif v =='D, C':
            color = "#EEAED4"
        elif v == 'D, D':
            color = "#8E0B52"
        return 'background-color: %s' % color


#colors for action_pairs plots 
reference = pd.DataFrame(['C, C', 'C, D', 'D, C', 'D, D'])
reference = reference.style.applymap(C_pairs_condition).set_caption(f"Colour map for action pairs")
#dfi.export(reference,"reference_color_map_for_action_pairs.png")


def plot_matrix_action_pairs(n_runs):
    '''visualise % mutual cooperation
    - consider only the final pair of actions to see where the pair ended up'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    figsize=(8, 6.5)
    types = ['S','UT','DE','Ve','Vk','Vm']

    matrix_CC = pd.DataFrame(columns=types, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_DD = pd.DataFrame(columns=types, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_DC = pd.DataFrame(columns=types, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_CD = pd.DataFrame(columns=types, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)


    for player1_title in ['QLS','QLUT','QLDE','QLVE_e','QLVE_k','QLVM']:
        action_pairs_against_QLS = pd.read_csv(f'results/{player1_title}_QLS/action_pairs.csv', index_col=0).iloc[9999]
        try:
            action_pairs_against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/action_pairs.csv', index_col=0).iloc[9999]
            order_QLUT = ''
        except:
            action_pairs_against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/action_pairs.csv', index_col=0).iloc[9999]
            order_QLUT = 'QLUT_first'
        try:
            action_pairs_against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/action_pairs.csv', index_col=0).iloc[9999]
            order_QLDE = ''
        except:
            action_pairs_against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/action_pairs.csv', index_col=0).iloc[9999]
            order_QLDE = 'QLDE_first'
        try:
            action_pairs_against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/action_pairs.csv', index_col=0).iloc[9999]
            order_QLVE_e = ''
        except:
            action_pairs_against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/action_pairs.csv', index_col=0).iloc[9999]
            order_QLVE_e = 'QLVE_e_first'
        try:
            action_pairs_against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/action_pairs.csv', index_col=0).iloc[9999]
            order_QLVE_k = ''
        except:
            action_pairs_against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/action_pairs.csv', index_col=0).iloc[9999]
            order_QLVE_k = 'QLVE_k_first'
        try:
            action_pairs_against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/action_pairs.csv', index_col=0).iloc[9999]
            order_QLVM = ''
        except:
            action_pairs_against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/action_pairs.csv', index_col=0).iloc[9999]
            order_QLVM = 'QLVM_first'

    
        action_pairs_against_QLS['%_CC'] = (action_pairs_against_QLS[action_pairs_against_QLS[:]=='C, C'].count() / n_runs)*100
        action_pairs_against_QLUT['%_CC'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='C, C'].count() / n_runs)*100
        action_pairs_against_QLDE['%_CC'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='C, C'].count() / n_runs)*100
        action_pairs_against_QLVE_e['%_CC'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='C, C'].count() / n_runs)*100
        action_pairs_against_QLVE_k['%_CC'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='C, C'].count() / n_runs)*100
        action_pairs_against_QLVM['%_CC'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='C, C'].count() / n_runs)*100

        short_title = player1_title.replace('QL','').replace('VE_e','Ve').replace('VE_k','Vk').replace('VM','Vm')
        matrix_CC.loc[short_title] = [action_pairs_against_QLS['%_CC'],action_pairs_against_QLUT['%_CC'],action_pairs_against_QLDE['%_CC'],action_pairs_against_QLVE_e['%_CC'],action_pairs_against_QLVE_k['%_CC'],action_pairs_against_QLVM['%_CC']]
        


        action_pairs_against_QLS['%_DD'] = (action_pairs_against_QLS[action_pairs_against_QLS[:]=='D, D'].count()/ n_runs)*100
        action_pairs_against_QLUT['%_DD'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='D, D'].count()/ n_runs)*100
        action_pairs_against_QLDE['%_DD'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='D, D'].count()/ n_runs)*100
        action_pairs_against_QLVE_e['%_DD'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='D, D'].count()/ n_runs)*100
        action_pairs_against_QLVE_k['%_DD'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='D, D'].count()/ n_runs)*100
        action_pairs_against_QLVM['%_DD'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='D, D'].count() / n_runs)*100

        matrix_DD.loc[short_title] = [action_pairs_against_QLS['%_DD'],action_pairs_against_QLUT['%_DD'],action_pairs_against_QLDE['%_DD'],action_pairs_against_QLVE_e['%_DD'],action_pairs_against_QLVE_k['%_DD'],action_pairs_against_QLVM['%_DD']]



        action_pairs_against_QLS['%_DC'] = (action_pairs_against_QLS[action_pairs_against_QLS[:]=='D, C'].count()/ n_runs)*100
        if order_QLUT=='QLUT_first':
            action_pairs_against_QLUT['%_DC'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='C, D'].count()/ n_runs)*100
        else: 
            action_pairs_against_QLUT['%_DC'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='D, C'].count()/ n_runs)*100
        if order_QLDE=='QLDE_first':
            action_pairs_against_QLDE['%_DC'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='C, D'].count()/ n_runs)*100
        else: 
            action_pairs_against_QLDE['%_DC'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='D, C'].count()/ n_runs)*100
        if order_QLVE_e=='QLVE_e_first':
            action_pairs_against_QLVE_e['%_DC'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='C, D'].count()/ n_runs)*100
        else: 
            action_pairs_against_QLVE_e['%_DC'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='D, C'].count()/ n_runs)*100
        if order_QLVE_k=='QLVE_k_first':
            action_pairs_against_QLVE_k['%_DC'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='C, D'].count()/ n_runs)*100
        else:
            action_pairs_against_QLVE_k['%_DC'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='D, C'].count()/ n_runs)*100
        if order_QLVM=='QLVM_first':
            action_pairs_against_QLVM['%_DC'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='C, D'].count()/ n_runs)*100
        else:
            action_pairs_against_QLVM['%_DC'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='D, C'].count()/ n_runs)*100
        
        matrix_DC.loc[short_title] = [action_pairs_against_QLS['%_DC'],action_pairs_against_QLUT['%_DC'],action_pairs_against_QLDE['%_DC'],action_pairs_against_QLVE_e['%_DC'],action_pairs_against_QLVE_k['%_DC'],action_pairs_against_QLVM['%_DC']]




        action_pairs_against_QLS['%_CD'] = (action_pairs_against_QLS[action_pairs_against_QLS[:]=='C, D'].count()/ n_runs)*100
        if order_QLUT=='QLUT_first':
            action_pairs_against_QLUT['%_CD'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='D, C'].count()/ n_runs)*100
        else: 
            action_pairs_against_QLUT['%_CD'] = (action_pairs_against_QLUT[action_pairs_against_QLUT[:]=='C, D'].count()/ n_runs)*100
        if order_QLDE=='QLDE_first':
            action_pairs_against_QLDE['%_CD'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='D, C'].count()/ n_runs)*100
        else: 
            action_pairs_against_QLDE['%_CD'] = (action_pairs_against_QLDE[action_pairs_against_QLDE[:]=='C, D'].count()/ n_runs)*100
        if order_QLVE_e=='QLVE_e_first':
            action_pairs_against_QLVE_e['%_CD'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='D, C'].count()/ n_runs)*100
        else: 
            action_pairs_against_QLVE_e['%_CD'] = (action_pairs_against_QLVE_e[action_pairs_against_QLVE_e[:]=='C, D'].count()/ n_runs)*100
        if order_QLVE_k=='QLVE_k_first':
            action_pairs_against_QLVE_k['%_CD'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='D, C'].count()/ n_runs)*100
        else:
            action_pairs_against_QLVE_k['%_CD'] = (action_pairs_against_QLVE_k[action_pairs_against_QLVE_k[:]=='C, D'].count()/ n_runs)*100
        if order_QLVM=='QLVM_first':
            action_pairs_against_QLVM['%_CD'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='D, C'].count()/ n_runs)*100
        else:
            action_pairs_against_QLVM['%_CD'] = (action_pairs_against_QLVM[action_pairs_against_QLVM[:]=='C, D'].count()/ n_runs)*100

        matrix_CD.loc[short_title] = [action_pairs_against_QLS['%_CD'],action_pairs_against_QLUT['%_CD'],action_pairs_against_QLDE['%_CD'],action_pairs_against_QLVE_e['%_CD'],action_pairs_against_QLVE_k['%_CD'],action_pairs_against_QLVM['%_CD']]



        #matrix_DCCD.loc[short_title] = action_pairs_against_QLS['%_DC'],action_pairs_against_QLUT['%_DC'],action_pairs_against_QLDE['%_DC'],action_pairs_against_QLVE_e['%_DC'],action_pairs_against_QLVE_k['%_DC']


    for label in types:
        for matrix in [matrix_CC, matrix_DD, matrix_DC, matrix_CD]:
            matrix[label] = matrix[label].astype(float)


    types_long = ['Selfish','Utilitarian','Deontolog.','Virtue-eq.','Virtue-kind.','Virtue-mix.']

    if not os.path.isdir('results/outcome_plots/actions_matrix'):
        os.makedirs('results/outcome_plots/actions_matrix')
        
    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_CC, cmap='YlGn', xticklabels=types_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = 0, vmax = 100)
    s.set(xlabel='Opponent O', ylabel='Player M',title='Mutual Cooperation \n')
    #plt.show()
    plt.savefig('results/outcome_plots/actions_matrix/CC.pdf', bbox_inches='tight')
    matrix_CC.to_csv('results/outcome_plots/actions_matrix/matrix_CC.csv')

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_DD, cmap='PuRd', xticklabels=types_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = 0, vmax = 100)
    s.set(xlabel='Opponent O', ylabel='Player M',title='Mutual Defection \n')
    #plt.show()
    plt.savefig('results/outcome_plots/actions_matrix/DD.pdf', bbox_inches='tight')
    matrix_DD.to_csv('results/outcome_plots/actions_matrix/matrix_DD.csv')

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_DC, cmap='Oranges', xticklabels=types_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = 0, vmax = 100)
    s.set(xlabel='Opponent O', ylabel='Player M',title='M Defects, O Cooperates\n ') #M exploits O
    #plt.show()
    plt.savefig('results/outcome_plots/actions_matrix/DC.pdf', bbox_inches='tight')
    matrix_DC.to_csv('results/outcome_plots/actions_matrix/matrix_DC.csv')

    plt.figure(figsize=figsize)
    s = sns.heatmap(matrix_CD, cmap='Blues', xticklabels=types_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = 0, vmax = 100)
    sns.set(font_scale=4)
    s.set(xlabel='Opponent O', ylabel='Player M', title='M Cooperates, O Defects\n') #M Getting Exploited
    #plt.show()
    plt.savefig('results/outcome_plots/actions_matrix/CD.pdf', bbox_inches='tight')
    matrix_CD.to_csv('results/outcome_plots/actions_matrix/matrix_CD.csv')

def plot_baseline_matrix_action_pairs(n_runs):
    '''visualise % mutual cooperation
    - consider only the final pair to see where the pair ended up'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    figsize=(8, 6.5)
    types = ['S','UT','DE','Ve','Vk','Vm']
    static_strategies = ['AC','AD','TFT','Random']

    matrix_CC = pd.DataFrame(columns=static_strategies, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_DD = pd.DataFrame(columns=static_strategies, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_DC = pd.DataFrame(columns=static_strategies, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_CD = pd.DataFrame(columns=static_strategies, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)


    for player1_title in ['QLS','QLUT','QLDE','QLVE_e','QLVE_k','QLVM']:
        action_pairs_against_AC = pd.read_csv(f'results/{player1_title}_AC/action_pairs.csv', index_col=0).iloc[9999]
        action_pairs_against_AD = pd.read_csv(f'results/{player1_title}_AD/action_pairs.csv', index_col=0).iloc[9999]
        action_pairs_against_TFT = pd.read_csv(f'results/{player1_title}_TFT/action_pairs.csv', index_col=0).iloc[9999]
        action_pairs_against_Random = pd.read_csv(f'results/{player1_title}_Random/action_pairs.csv', index_col=0).iloc[9999]


        action_pairs_against_AC['%_CC'] = (action_pairs_against_AC[action_pairs_against_AC[:]=='C, C'].count() / n_runs)*100
        action_pairs_against_AD['%_CC'] = (action_pairs_against_AD[action_pairs_against_AD[:]=='C, C'].count() / n_runs)*100
        action_pairs_against_TFT['%_CC'] = (action_pairs_against_TFT[action_pairs_against_TFT[:]=='C, C'].count() / n_runs)*100
        action_pairs_against_Random['%_CC'] = (action_pairs_against_Random[action_pairs_against_Random[:]=='C, C'].count() / n_runs)*100


        short_title = player1_title.replace('QL','').replace('VE_e','Ve').replace('VE_k','Vk').replace('VM','Vm')
        matrix_CC.loc[short_title] = [action_pairs_against_AC['%_CC'],action_pairs_against_AD['%_CC'],action_pairs_against_TFT['%_CC'],action_pairs_against_Random['%_CC']]
        

        action_pairs_against_AC['%_DD'] = (action_pairs_against_AC[action_pairs_against_AC[:]=='D, D'].count() / n_runs)*100
        action_pairs_against_AD['%_DD'] = (action_pairs_against_AD[action_pairs_against_AD[:]=='D, D'].count() / n_runs)*100
        action_pairs_against_TFT['%_DD'] = (action_pairs_against_TFT[action_pairs_against_TFT[:]=='D, D'].count() / n_runs)*100
        action_pairs_against_Random['%_DD'] = (action_pairs_against_Random[action_pairs_against_Random[:]=='D, D'].count() / n_runs)*100

        matrix_DD.loc[short_title] = [action_pairs_against_AC['%_DD'],action_pairs_against_AD['%_DD'],action_pairs_against_TFT['%_DD'],action_pairs_against_Random['%_DD']]


        action_pairs_against_AC['%_DC'] = (action_pairs_against_AC[action_pairs_against_AC[:]=='D, C'].count() / n_runs)*100
        action_pairs_against_AD['%_DC'] = (action_pairs_against_AD[action_pairs_against_AD[:]=='D, C'].count() / n_runs)*100
        action_pairs_against_TFT['%_DC'] = (action_pairs_against_TFT[action_pairs_against_TFT[:]=='D, C'].count() / n_runs)*100
        action_pairs_against_Random['%_DC'] = (action_pairs_against_Random[action_pairs_against_Random[:]=='D, C'].count() / n_runs)*100
        
        matrix_DC.loc[short_title] = [action_pairs_against_AC['%_DC'],action_pairs_against_AD['%_DC'],action_pairs_against_TFT['%_DC'],action_pairs_against_Random['%_DC']]


        action_pairs_against_AC['%_CD'] = (action_pairs_against_AC[action_pairs_against_AC[:]=='C, D'].count() / n_runs)*100
        action_pairs_against_AD['%_CD'] = (action_pairs_against_AD[action_pairs_against_AD[:]=='C, D'].count() / n_runs)*100
        action_pairs_against_TFT['%_CD'] = (action_pairs_against_TFT[action_pairs_against_TFT[:]=='C, D'].count() / n_runs)*100
        action_pairs_against_Random['%_CD'] = (action_pairs_against_Random[action_pairs_against_Random[:]=='C, D'].count() / n_runs)*100
        
        matrix_CD.loc[short_title] = [action_pairs_against_AC['%_CD'],action_pairs_against_AD['%_CD'],action_pairs_against_TFT['%_CD'],action_pairs_against_Random['%_CD']]

        #matrix_DCCD.loc[short_title] = action_pairs_against_QLS['%_DC'],action_pairs_against_QLUT['%_DC'],action_pairs_against_QLDE['%_DC'],action_pairs_against_QLVE_e['%_DC'],action_pairs_against_QLVE_k['%_DC']


    for label in static_strategies:
        for matrix in [matrix_CC, matrix_DD, matrix_DC, matrix_CD]:
            matrix[label] = matrix[label].astype(float)


    types_long = ['Selfish','Utilitarian','Deontolog.','Virtue-eq.','Virtue-kind.','Virtue-mix.']
    static_strategies_long = ['Always Coop.','Always Defect','Tit for Tat','Random']

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_CC, cmap='YlGn', xticklabels=static_strategies_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = 0, vmax = 100)
    s.set(xlabel='Opponent O', ylabel='Player M',title='Mutual Cooperation \n')
    #plt.show()
    plt.savefig('results/outcome_plots/actions_matrix/baseline_CC.pdf', bbox_inches='tight')
    matrix_CC.to_csv('results/outcome_plots/actions_matrix/baseline_matrix_CC.csv')


    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_DD, cmap='PuRd', xticklabels=static_strategies_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = 0, vmax = 100)
    s.set(xlabel='Opponent O', ylabel='Player M',title='Mutual Defection \n')
    #plt.show()
    plt.savefig('results/outcome_plots/actions_matrix/baseline_DD.pdf', bbox_inches='tight')
    matrix_DD.to_csv('results/outcome_plots/actions_matrix/baseline_matrix_DD.csv')


    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_DC, cmap='Oranges', xticklabels=static_strategies_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = 0, vmax = 100)
    s.set(xlabel='Opponent O', ylabel='Player M',title='M Defects, O Cooperates\n ') #M exploits O
    #plt.show()
    plt.savefig('results/outcome_plots/actions_matrix/baseline_DC.pdf', bbox_inches='tight')
    matrix_DC.to_csv('results/outcome_plots/actions_matrix/baseline_matrix_DC.csv')


    plt.figure(figsize=figsize)
    s = sns.heatmap(matrix_CD, cmap='Blues', xticklabels=static_strategies_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = 0, vmax = 100)
    sns.set(font_scale=4)
    s.set(xlabel='Opponent O', ylabel='Player M', title='M Cooperates, O Defects \n') #M Getting Exploited
    #plt.show()
    plt.savefig('results/outcome_plots/actions_matrix/baseline_CD.pdf', bbox_inches='tight')
    matrix_CD.to_csv('results/outcome_plots/actions_matrix/baseline_matrix_CD.csv')


def plot_matrix_social_outcomes(n_runs):
    '''visualise % mutual cooperation
    - consider only the final pair to see where the pair ended up'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    figsize=(8, 6.5)
    types = ['S','UT','DE','Ve','Vk','Vm']
    types_long = ['Selfish','Utilitarian','Deontolog.','Virtue-eq.','Virtue-kind.','Virtue-mix.']

    matrix_collective = pd.DataFrame(columns=types, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_gini = pd.DataFrame(columns=types, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_min = pd.DataFrame(columns=types, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)

    ylabs = {}
    ylims = {}

    if not os.path.isdir('results/outcome_plots/outcomes_matrix'):
        os.makedirs('results/outcome_plots/outcomes_matrix')

    for player1_title in ['QLS','QLUT','QLDE','QLVE_e','QLVE_k','QLVM']:
        short_title = player1_title.replace('QL','').replace('VE_e','Ve').replace('VE_k','Vk').replace('VM','Vm')
        
        for type in ['collective', 'gini', 'min']:
            against_QLS = pd.read_csv(f'results/{player1_title}_QLS/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_QLS_mean = against_QLS.mean()
            against_QLS_sds = against_QLS.std()
            against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

            try:
                against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            except:
                against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_QLUT_mean = against_QLUT.mean()
            against_QLUT_sds = against_QLUT.std()
            against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

            try:
                against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            except:
                against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_QLDE_mean = against_QLDE.mean()
            against_QLDE_sds = against_QLDE.std()
            against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

            try:
                against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            except:
                against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_QLVE_e_mean = against_QLVE_e.mean()
            against_QLVE_e_sds = against_QLVE_e.std()
            against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

            try:
                against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            except:
                against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_QLVE_k_mean = against_QLVE_k.mean()
            against_QLVE_k_sds = against_QLVE_k.std()
            against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

            try:
                against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            except:
                against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_QLVM_mean = against_QLVM.mean()
            against_QLVM_sds = against_QLVM.std()
            against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)
        
            ylabs[f'{type}'] = (f'$G_{{{type}}}$')

            if game_title=='IPD':
                if type=='collective':
                    ylims[f'{type}'] = [40000, 60000]
                elif type=='gini':
                    ylims[f'{type}'] = [5000, 10000] #5714.285714285714 min vale for G_gini
                elif type=='min':
                    ylims[f'{type}'] = [10000, 30000]
            elif game_title=='VOLUNTEER':
                if type=='collective': 
                    ylims[f'{type}'] = [20000, 80000]
                elif type=='gini':
                    ylims[f'{type}'] = [4000, 10000]
                elif type=='min':
                    ylims[f'{type}'] = [10000, 40000]
            elif game_title=='STAGHUNT': 
                if type=='collective': 
                    ylims[f'{type}'] = [40000, 100000]
                elif type=='gini':
                    ylims[f'{type}'] = [4000, 10000]
                elif type=='min':
                    ylims[f'{type}'] = [10000, 40000]
            


            if type == 'collective': 
                matrix_collective.loc[short_title] = [against_QLS_mean,against_QLUT_mean,against_QLDE_mean, against_QLVE_e_mean, against_QLVE_k_mean, against_QLVM_mean]
                for label in types:
                    matrix_collective[label] = matrix_collective[label].astype(float)

            elif type == 'gini':
                matrix_gini.loc[short_title] = [against_QLS_mean,against_QLUT_mean,against_QLDE_mean, against_QLVE_e_mean, against_QLVE_k_mean, against_QLVM_mean]
                for label in types:
                    matrix_gini[label] = matrix_gini[label].astype(float)
            
            elif type == 'min':
                matrix_min.loc[short_title] = [against_QLS_mean,against_QLUT_mean,against_QLDE_mean, against_QLVE_e_mean, against_QLVE_k_mean, against_QLVM_mean]
                for label in types:
                    matrix_min[label] = matrix_min[label].astype(float)

                
    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_collective, cmap='BrBG', xticklabels=types_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = ylims['collective'][0], vmax = ylims['collective'][1])
    s.set(xlabel='Opponent O', ylabel='Player M',title='Collective Return '+ylabs['collective']+' \n')
    plt.savefig('results/outcome_plots/outcomes_matrix/collective.pdf', bbox_inches='tight')
    matrix_collective.to_csv('results/outcome_plots/outcomes_matrix/matrix_collective.csv')

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_gini, cmap='BrBG', xticklabels=types_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = ylims['gini'][0], vmax = ylims['gini'][1])
    s.set(xlabel='Opponent O', ylabel='Player M',title='Gini Return '+ylabs['gini']+' \n')
    plt.savefig('results/outcome_plots/outcomes_matrix/gini.pdf', bbox_inches='tight')
    matrix_gini.to_csv('results/outcome_plots/outcomes_matrix/matrix_gini.csv')

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_min, cmap='BrBG', xticklabels=types_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = ylims['min'][0], vmax = ylims['min'][1])
    s.set(xlabel='Opponent O', ylabel='Player M',title='Minimum Return '+ylabs['min']+' \n')
    plt.savefig('results/outcome_plots/outcomes_matrix/min.pdf', bbox_inches='tight')
    matrix_min.to_csv('results/outcome_plots/outcomes_matrix/matrix_min.csv')
        
def plot_baseline_matrix_social_outcomes(n_runs):
    '''visualise % mutual cooperation
    - consider only the final pair to see where the pair ended up'''
    #NOTE this will only work for 10000 iterations right now, not fewer!!!
    #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

    figsize=(8, 6.5)
    types = ['S','UT','DE','Ve','Vk','Vm']
    static_strategies = ['AC','AD','TFT','Random']
    types_long = ['Selfish','Utilitarian','Deontolog.','Virtue-eq.','Virtue-kind.','Virtue-mix.']
    static_strategies_long = ['Always Coop.','Always Defect','Tit for Tat','Random']

    matrix_collective = pd.DataFrame(columns=static_strategies, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_gini = pd.DataFrame(columns=static_strategies, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)
    matrix_min = pd.DataFrame(columns=static_strategies, index=types) #shape = vs S,UT,DE,Ve,Vk(,VEm)

    ylabs = {}
    ylims = {}

    if not os.path.isdir('results/outcome_plots/outcomes_matrix'):
        os.makedirs('results/outcome_plots/outcomes_matrix')

    for player1_title in ['QLS','QLUT','QLDE','QLVE_e','QLVE_k','QLVM']:
        short_title = player1_title.replace('QL','').replace('VE_e','Ve').replace('VE_k','Vk').replace('VM','Vm')
        
        for type in ['collective', 'gini', 'min']:
            against_AC = pd.read_csv(f'results/{player1_title}_AC/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_AC_mean = against_AC.mean()
            against_AC_sds = against_AC.std()
            against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

            against_AD = pd.read_csv(f'results/{player1_title}_AD/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_AD_mean = against_AD.mean()
            against_AD_sds = against_AD.std()
            against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

            against_TFT = pd.read_csv(f'results/{player1_title}_TFT/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_TFT_mean = against_TFT.mean()
            against_TFT_sds = against_TFT.std()
            against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

            against_Random = pd.read_csv(f'results/{player1_title}_Random/df_cumulative_reward_{type}.csv', index_col=0).iloc[-1]
            against_Random_mean = against_Random.mean()
            against_Random_sds = against_Random.std()
            against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

        
            ylabs[f'{type}'] = (f'$G_{{{type}}}$')

            if game_title=='IPD':
                if type=='collective':
                    ylims[f'{type}'] = [40000, 60000]
                elif type=='gini':
                    ylims[f'{type}'] = [5000, 10000] #5714.285714285714 min vale for Gmin
                elif type=='min':
                    ylims[f'{type}'] = [10000, 30000]
            elif game_title=='VOLUNTEER':
                if type=='collective': 
                    ylims[f'{type}'] = [20000, 80000]
                elif type=='gini':
                    ylims[f'{type}'] = [4000, 10000]
                elif type=='min':
                    ylims[f'{type}'] = [10000, 40000]
            elif game_title=='STAGHUNT': 
                if type=='collective': 
                    ylims[f'{type}'] = [40000, 100000]
                elif type=='gini':
                    ylims[f'{type}'] = [4000, 10000]
                elif type=='min':
                    ylims[f'{type}'] = [10000, 40000]
            

            if type == 'collective': 
                matrix_collective.loc[short_title] = [against_AC_mean, against_AD_mean, against_TFT_mean, against_Random_mean]
                for label in static_strategies:
                    matrix_collective[label] = matrix_collective[label].astype(float)

            elif type == 'gini':
                matrix_gini.loc[short_title] = [against_AC_mean, against_AD_mean, against_TFT_mean, against_Random_mean]
                for label in static_strategies:
                    matrix_gini[label] = matrix_gini[label].astype(float)
            
            elif type == 'min':
                matrix_min.loc[short_title] = [against_AC_mean, against_AD_mean, against_TFT_mean, against_Random_mean]
                for label in static_strategies:
                    matrix_min[label] = matrix_min[label].astype(float)

                
    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_collective, cmap='BrBG', xticklabels=static_strategies_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = ylims['collective'][0], vmax = ylims['collective'][1])
    s.set(xlabel='Opponent O', ylabel='Player M',title='Collective Return '+ylabs['collective']+' \n')
    plt.savefig('results/outcome_plots/outcomes_matrix/baseline_collective.pdf', bbox_inches='tight')
    matrix_collective.to_csv('results/outcome_plots/outcomes_matrix/baseline_matrix_collective.csv')

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_gini, cmap='BrBG', xticklabels=static_strategies_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = ylims['gini'][0], vmax = ylims['gini'][1])
    s.set(xlabel='Opponent O', ylabel='Player M',title='Gini Return '+ylabs['gini']+' \n')
    plt.savefig('results/outcome_plots/outcomes_matrix/baseline_gini.pdf', bbox_inches='tight')
    matrix_gini.to_csv('results/outcome_plots/outcomes_matrix/baseline_matrix_gini.csv')

    plt.figure(figsize=figsize)
    sns.set(font_scale=4)
    s = sns.heatmap(matrix_min, cmap='BrBG', xticklabels=static_strategies_long, yticklabels=types_long, annot=False, linecolor='black', linewidths=0, square=False, vmin = ylims['min'][0], vmax = ylims['min'][1])
    s.set(xlabel='Opponent O', ylabel='Player M',title='Minimum Return '+ylabs['min']+' \n')
    plt.savefig('results/outcome_plots/outcomes_matrix/baseline_min.pdf', bbox_inches='tight')
    matrix_min.to_csv('results/outcome_plots/outcomes_matrix/baseline_matrix_min.csv')
            

def plot_stacked_relative_reward(n_runs):
    '''plot game reward - relatie cumulative (bar & over time) & per iteration
    - how well off did the players end up relative to each other on the game?'''

    data = dict() #keys=['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k', 'QLVM'], values=[]

    for player1_title in ['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k', 'QLVM']: 
        ##################################
        #### bar chart game cumulative reward for player1_tytle vs others  ####
        ##################################
        against_QLS = pd.read_csv(f'results/{player1_title}_QLS/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_QLS_means = against_QLS.mean()
        against_QLS_sds = against_QLS.std()
        against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

        try:
            against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        except:
            against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_QLUT_means = against_QLUT.mean()
        against_QLUT_sds = against_QLUT.std()
        against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

        try:
            against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        except:
            against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_QLDE_means = against_QLDE.mean()
        against_QLDE_sds = against_QLDE.std()
        against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

        try:
            against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        except:
            against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_QLVE_e_means = against_QLVE_e.mean()
        against_QLVE_e_sds = against_QLVE_e.std()
        against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

        try:
            against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        except:
            against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_QLVE_k_means = against_QLVE_k.mean()
        against_QLVE_k_sds = against_QLVE_k.std()
        against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

        try:
            against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        except:
            against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/player2/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_QLVM_means = against_QLVM.mean()
        against_QLVM_sds = against_QLVM.std()
        against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)

        data[str('means_'+player1_title)] = [against_QLS_means, against_QLUT_means, against_QLDE_means, against_QLVE_e_means, against_QLVE_k_means, against_QLVM_means]
        data[str('CIs_'+player1_title)] = [against_QLS_ci, against_QLUT_ci, against_QLDE_ci, against_QLVE_e_ci, against_QLVE_k_ci, against_QLVM_ci]



    labels = ['S', 'UT', 'DE', 'V'+r'$_e$', 'V'+r'$_k$', 'V'+r'$_m$'] 
    #colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple', 'pink'] 
    colors = ['#984464', '#e6a176', '#556b2f', '#5eccab', '#537eff', '#c0affb']
    #colors = ['#e1562c', 'orange', '#556b2f', 'lightblue', '#00678a', 'pink']
    #plt.rcParams.update({'font.size':20})
    font = {'size'   : 15}
    matplotlib.rc('font', **font)

    fig, axes = plt.subplots(1, 6, figsize=(11, 3.5), sharey=True) #constrained_layout=False, tight_layout=False, 
    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    ax1.bar(labels, data['means_QLS'], yerr=data['CIs_QLS'], color=colors) #, width = 0.8
    ax1.set_title('Selfish') #vs others
    ax1.set_ylabel('Cumulative \n' + 'Game Reward') #r'$R_{extr}$'
    #ax1.set_ylim()
    if game_title=='IPD': #NOTE game_title is set outside this function - in the overall environment - see code below 
        plt.gca().set_ylim([0, 40000])
    elif game_title=='VOLUNTEER':
        plt.gca().set_ylim([0, 50000])
    elif game_title=='STAGHUNT': 
        plt.gca().set_ylim([0, 50000])  
    #ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45)
    #ax1.set_xticklabels(labels, rotation=90, ha='right')
    #ax1.yaxis.set_label(r'Cumulative Game Reward $R_{extr}$')
 
    
    ax2.bar(labels, data['means_QLUT'], yerr=data['CIs_QLUT'], color=colors) #, width = 0.8
    ax2.set_title('Utilitarian')

    ax3.bar(labels, data['means_QLDE'], yerr=data['CIs_QLDE'], color=colors) #, width = 0.8
    ax3.set_title('Deontolog.')

    ax4.bar(labels, data['means_QLVE_e'], yerr=data['CIs_QLVE_e'], color=colors) #, width = 0.8
    ax4.set_title(' Virtue-equal.')

    ax5.bar(labels, data['means_QLVE_k'], yerr=data['CIs_QLVE_k'], color=colors) #, width = 0.8
    ax5.set_title(' Virtue-kind.')

    ax6.bar(labels, data['means_QLVM'], yerr=data['CIs_QLVM'], color=colors) #, width = 0.8
    ax6.set_title(' Virtue-mixed')

    fig.suptitle('Game Reward for player type M vs all others \n')
    
    #ax1.yaxis.set_label(r'Cumulative Game Reward $R_{extr}$')
    fig.autofmt_xdate(rotation=90, bottom=0.2, ha='center')
    #plt.xticks(rotation=45)
    fig.supxlabel('\n'+'Opponent type') #NOTE this requires matplotlib>3.4 
    plt.tight_layout(pad=0.001, w_pad=0.8)
    fig.show()


    if not os.path.isdir('results/outcome_plots/reward'):
        os.makedirs('results/outcome_plots/reward')
    plt.savefig(f'results/outcome_plots/reward/new_cumulative_game_reward.png', bbox_inches='tight')
    

def set_ylims(game_title, player1_title): 
    '''this function sets ylims for the plots of cumulative moral reward across all 3 games & 6 player types'''
    ylims = [] 
    if game_title=='IPD':
        if player1_title == 'QLUT':
            ylims[0:1] = [0, 60000]
        elif player1_title == 'QLDE':
            ylims[0:1] = [-50000, 0]
        elif player1_title == 'QLVE_e':
            ylims[0:1] = [0, 10000]
        elif player1_title == 'QLVE_k':
            ylims[0:1] = [0, 50000]
        elif player1_title == 'QLVM':
            ylims[0:1] = [0, 10000]
    elif game_title=='VOLUNTEER':
        if player1_title == 'QLUT':
            ylims[0:1] = [0, 80000]
        elif player1_title == 'QLDE':
            ylims[0:1] = [-50000, 0]
        elif player1_title == 'QLVE_e':
            ylims[0:1] = [0, 10000]
        elif player1_title == 'QLVE_k':
            ylims[0:1] = [0, 50000]
        elif player1_title == 'QLVM':
            ylims[0:1] = [0, 10000]
    elif game_title=='STAGHUNT': 
        if player1_title == 'QLUT':
            ylims[0:2] = [0, 100000]
        elif player1_title == 'QLDE':
            ylims[0:2] = [-50000, 0]
        elif player1_title == 'QLVE_e':
            ylims[0:2] = [0, 10000]
        elif player1_title == 'QLVE_k':
            ylims[0:2] = [0, 50000]
        elif player1_title == 'QLVM':
            ylims[0:2] = [0, 10000]
    return ylims[0], ylims[1]

def plot_stacked_relative_moral_reward(n_runs):
    '''plot game reward - relatie cumulative (bar & over time) & per iteration
    - how well off did the players end up relative to each other on the game?'''

    data = dict() #keys=['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k', 'QLVM'], values=[]

    for player1_title in ['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k', 'QLVM']: 
        ##################################
        #### bar chart game cumulative reward for player1_tytle vs others  ####
        ##################################
        against_QLS = pd.read_csv(f'results/{player1_title}_QLS/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_QLS_means = against_QLS.mean()
        against_QLS_sds = against_QLS.std()
        against_QLS_ci = 1.96 * against_QLS_sds/np.sqrt(n_runs)

        try:
            against_QLUT = pd.read_csv(f'results/{player1_title}_QLUT/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        except:
            against_QLUT = pd.read_csv(f'results/QLUT_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_QLUT_means = against_QLUT.mean()
        against_QLUT_sds = against_QLUT.std()
        against_QLUT_ci = 1.96 * against_QLUT_sds/np.sqrt(n_runs)

        try:
            against_QLDE = pd.read_csv(f'results/{player1_title}_QLDE/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        except:
            against_QLDE = pd.read_csv(f'results/QLDE_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_QLDE_means = against_QLDE.mean()
        against_QLDE_sds = against_QLDE.std()
        against_QLDE_ci = 1.96 * against_QLDE_sds/np.sqrt(n_runs)

        try:
            against_QLVE_e = pd.read_csv(f'results/{player1_title}_QLVE_e/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        except:
            against_QLVE_e = pd.read_csv(f'results/QLVE_e_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_QLVE_e_means = against_QLVE_e.mean()
        against_QLVE_e_sds = against_QLVE_e.std()
        against_QLVE_e_ci = 1.96 * against_QLVE_e_sds/np.sqrt(n_runs)

        try:
            against_QLVE_k = pd.read_csv(f'results/{player1_title}_QLVE_k/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        except:
            against_QLVE_k = pd.read_csv(f'results/QLVE_k_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_QLVE_k_means = against_QLVE_k.mean()
        against_QLVE_k_sds = against_QLVE_k.std()
        against_QLVE_k_ci = 1.96 * against_QLVE_k_sds/np.sqrt(n_runs)

        try:
            against_QLVM = pd.read_csv(f'results/{player1_title}_QLVM/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        except:
            against_QLVM = pd.read_csv(f'results/QLVM_{player1_title}/player2/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_QLVM_means = against_QLVM.mean()
        against_QLVM_sds = against_QLVM.std()
        against_QLVM_ci = 1.96 * against_QLVM_sds/np.sqrt(n_runs)

        data[str('means_'+player1_title)] = [against_QLS_means, against_QLUT_means, against_QLDE_means, against_QLVE_e_means, against_QLVE_k_means, against_QLVM_means]
        data[str('CIs_'+player1_title)] = [against_QLS_ci, against_QLUT_ci, against_QLDE_ci, against_QLVE_e_ci, against_QLVE_k_ci, against_QLVM_ci]



    labels = ['S', 'UT', 'DE', 'V'+r'$_e$', 'V'+r'$_k$', 'V'+r'$_m$'] 
    #colors = ['red', '#556b2f', '#00cccc', 'orange', 'purple', 'pink'] 
    colors = ['#984464', '#e6a176', '#556b2f', '#5eccab', '#537eff', '#c0affb']
    #colors = ['#e1562c', 'orange', '#556b2f', 'lightblue', '#00678a', 'pink']
    #plt.rcParams.update({'font.size':20})
    font = {'size'   : 15}
    matplotlib.rc('font', **font)

    fig, axes = plt.subplots(1, 5, figsize=(11, 3.5), sharey=False) #constrained_layout=False, tight_layout=False, 
    ax2, ax3, ax4, ax5, ax6 = axes

    #ax1.bar(labels, data['means_QLS'], yerr=data['CIs_QLS'], color=colors) #, width = 0.8
    #ax1.set_title('Selfish') #vs others
    #ax1.set_ylabel('Cumulative \n' + 'Moral Reward') #r'$R_{extr}$' 
    
    ax2.bar(labels, data['means_QLUT'], yerr=data['CIs_QLUT'], color=colors) #, width = 0.8
    ax2.set_title('Utilitarian')
    ylims = set_ylims(game_title, 'QLUT')
    ax2.set_ylim(ylims)

    ax3.bar(labels, data['means_QLDE'], yerr=data['CIs_QLDE'], color=colors) #, width = 0.8
    ax3.set_title('Deontolog.')
    ylims = set_ylims(game_title, 'QLDE')
    ax3.set_ylim(ylims)

    ax4.bar(labels, data['means_QLVE_e'], yerr=data['CIs_QLVE_e'], color=colors) #, width = 0.8
    ax4.set_title(' Virtue-equal.')
    ylims = set_ylims(game_title, 'QLVE_e')
    ax4.set_ylim(ylims)

    ax5.bar(labels, data['means_QLVE_k'], yerr=data['CIs_QLVE_k'], color=colors) #, width = 0.8
    ax5.set_title(' Virtue-kind.')
    ylims = set_ylims(game_title, 'QLVE_k')
    ax5.set_ylim(ylims)

    ax6.bar(labels, data['means_QLVM'], yerr=data['CIs_QLVM'], color=colors) #, width = 0.8
    ax6.set_title(' Virtue-mixed')
    ylims = set_ylims(game_title, 'QLVM')
    ax6.set_ylim(ylims)

    for ax in axes:
        for label in ax.get_yticklabels():
	        label.set_fontsize(11)

    fig.suptitle('Moral Reward for player type M vs all others \n')
    
    ax2.set_ylabel('Cumulative \n Moral Reward')
    fig.autofmt_xdate(rotation=90, bottom=0.2, ha='center')
    #plt.xticks(rotation=45)
    fig.supxlabel('\n'+'Opponent type') #NOTE this requires matplotlib>3.4 
    plt.tight_layout(pad=0.001, w_pad=0.1)
    fig.show()


    if not os.path.isdir('results/outcome_plots/reward'):
        os.makedirs('results/outcome_plots/reward')
    plt.savefig(f'results/outcome_plots/reward/new_cumulative_intrinsic_reward.png', bbox_inches='tight')


def analyse_cooperative_selections(destination_folder, titles, n_runs, num_iter, iter_range=None):
    '''create cooperative_selection bool variable for each run, then plot % cooperative selections'''

    colnames = ['run'+str(i+1) for i in range(n_runs)]

    data = pd.DataFrame(columns=colnames)
    if iter_range: 
        results_idx = pd.DataFrame(index=list(range(iter_range[0], iter_range[1])))
    else: 
        results_idx = pd.DataFrame(index=list(range(num_iter)))
    
    for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv')
        run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
        if iter_range: 
            run_df = run_df.iloc[iter_range[0]:iter_range[1]]
        run_df['cooperative_selection'] = run_df['selected_prev_move']==0 #.apply(lambda x: int(x))
        data[run_name] = run_df['cooperative_selection']

    results_idx['cooperative_selection'] = data[data[:]==True].count(axis='columns') / data.count(axis='columns') * 100

    return data, results_idx

def analyse_reasons_fillingbuffer(destination_folder, titles, n_runs, num_iter, iter_range=None):

    colnames = ['run'+str(i+1) for i in range(n_runs)]

    data = pd.DataFrame(columns=colnames)
    if iter_range: 
        results_idx = pd.DataFrame(index=list(range(iter_range[0], iter_range[1])))
    else: 
        results_idx = pd.DataFrame(index=list(range(num_iter)))
    
    for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv')
        run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
        if iter_range: 
            run_df = run_df.iloc[iter_range[0]:iter_range[1]]
        run_df['reason_random_fillingbuffer'] = run_df['reason_selection_player1']=='random, due to non-full memory buffer for selection' 
        data[run_name] = run_df['reason_random_fillingbuffer']

    results_idx['reason_random_fillingbuffer'] = data[data[:]==True].count(axis='columns') / data.count(axis='columns') * 100

    return data, results_idx

def analyse_reasons_eps(destination_folder, titles, n_runs, num_iter, iter_range=None):

    colnames = ['run'+str(i+1) for i in range(n_runs)]

    data = pd.DataFrame(columns=colnames)
    if iter_range: 
        results_idx = pd.DataFrame(index=list(range(iter_range[0], iter_range[1])))
    else: 
        results_idx = pd.DataFrame(index=list(range(num_iter)))
    
    for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv')
        run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
        if iter_range: 
            run_df = run_df.iloc[iter_range[0]:iter_range[1]]
        run_df['reason_random_eps'] = run_df['reason_selection_player1']=='random, due to eps in selection' 
        data[run_name] = run_df['reason_random_eps']

    results_idx['reason_random_eps'] = data[data[:]==True].count(axis='columns') / data.count(axis='columns') * 100

    return data, results_idx


#custom functions for population of 3 - S,AC,AD
def plot_opponents_selected(destination_folder, n_runs, palette, order, long_titles, title='Selfish', idx=None, episode_range=None): #opponent_titles
    '''separately for each run, plot the opponents selected by Player1==title 
    NOTE this is a custom function for the piopulation S,AC,AD '''
    if n_runs == 1:
        fig, axs =  plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True,figsize=(3,3))
    elif n_runs == 4: 
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(20,2.5))
    elif n_runs == 5:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, figsize=(20,2.5))
    elif n_runs == 7:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(nrows=1, ncols=7, sharey=True, figsize=(30,2.5))
    elif n_runs == 9:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(nrows=1, ncols=9, sharey=True, figsize=(35,2.5)) 
    elif n_runs == 10:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, figsize=(28,2.5)) 
    elif n_runs == 15: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15) = plt.subplots(nrows=1, ncols=15, sharey=True, figsize=(20,2.5))
    #elif n_runs == 20: 
    #    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(40,2.5))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=n_runs, sharey=True, figsize=(35,2.5)) 
    #overwrite the above 
    #fig, axs = plt.subplots(nrows=1, ncols=n_runs, sharey=True, figsize=(35,2.5)) 


    if episode_range:
        episode_idxs = list(range(episode_range[0],episode_range[1]))
        episode_range_str = f'episodes {episode_range[0]} to {episode_range[1]}'
    else:
        episode_idxs = list(range(1,num_iter))
        episode_range_str = f'episodes 1 to {num_iter}'
        #Nnote num_iter is defined globally 

    if idx==None: 
        count_players_with_title = long_titles.count(title)
        plt.suptitle(f'No of times each opponent was selected by {title} players (x{count_players_with_title}) \n ', y=1.1, fontsize=18)
    else:
        print(f'plotting for idx {idx} only')
        plt.suptitle(f'Selections by {title} player {idx} \n {episode_range_str} \n', y=1.1, fontsize=18)
    

    for run in range(1,n_runs+1):
        print(f'run {run}')
        run_df_unfiltered = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)
        run_df = run_df_unfiltered[run_df_unfiltered['episode'].isin(episode_idxs)]
    
        #ax = locals()["ax"+str(run)]
        if n_runs == 1:
            ax = axs
        else: 
            ax = axs[run-1]
            ax.set_title(f'\n run{run}') #No of times each action was played 

        if idx==None: #if plotting for allp layers with this title / type - NOTE need to fix opponents in this case! 
            small_run_df = run_df[run_df['title_p1']==title]
        else:
            small_run_df = run_df[(run_df['title_p1'] == title) & (run_df['idx_p1']==idx)]
        
        small_run_df['label_p2'] = small_run_df['title_p2'] + small_run_df['idx_p2'].astype(str)

        if False: 
            #opponent_mapping = {1:opponent_titles[0], 2:opponent_titles[1]}
            opponent_mapping = {}
            keys = range(len(opponent_titles))
            values = opponent_titles
            for i in keys:
                    opponent_mapping[i+1] = values[i]
            
            small_run_df['idx_p2'] = small_run_df['idx_p2'].replace(opponent_mapping)
            #small_run_df.loc[:,'idx_p2'] = small_run_df.loc[:,'idx_p2'].replace({1:opponent_titles[0], 2:opponent_titles[1]})
            #small_run_df['idx_p2'].replace({1:opponent_titles[0], 2:opponent_titles[1]}, inplace=True)
            #small_run_df.action_player1 = pd.Categorical(small_run_df.action_player1, categories=['C', 'D'], ordered=True)

            #palette = [color_mapping[title.split('_')[0]] for title in opponent_titles] #['green','green','green','green', 'orchid','orchid','orchid','orchid','orchid']
            palette = [color_mapping[title] for title in opponent_titles] #['green','green','green','green', 'orchid','orchid','orchid','orchid','orchid']

            chart = sns.countplot(ax=ax, data=small_run_df, x=small_run_df['idx_p2'], order=opponent_titles, palette=palette)
            #sns.set(font_scale = 0.5)
            chart.set_xticklabels(
            chart.get_xticklabels(), 
            rotation=90, 
            horizontalalignment='center',
            fontweight='light',
            fontsize='xx-small'
            )

            chart.set(xlabel='')

        chart = sns.countplot(ax=ax, data=small_run_df, x=small_run_df['label_p2'], order=order, palette=palette)
        #sns.set(font_scale = 0.5)
        chart.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=90, 
        horizontalalignment='center',
        fontweight='light',
        fontsize='xx-small'
        )
        #chart.set_title(f'Selected by {title} players')
        chart.set(xlabel='')

    if False: 
        print(f'run {run}')
        run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)

        counts = pd.DataFrame(run_df[run_df['title_p1']=='title']['idx_p2'].value_counts())
        print(counts)
        counts.rename({1:'AC', 2:'AD'}, axis=0, inplace=True)
        counts.loc[['AC', 'AD']].plot(kind='bar', title=f'No of times each opponent was selected \n run{run}', legend=False) 

    plt.xlabel('selection_player1') #to save space 

def plot_opponents_selected_last100(destination_folder, n_runs, palette, order, long_titles, title='Selfish', idx=None): #opponent_titles
    '''separately for each run, plot the opponents selected by Player1==title 
    NOTE this is a custom function for the piopulation S,AC,AD '''
    if n_runs == 1:
        fig, (ax1) =  plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True,figsize=(10,2.5))
    elif n_runs == 4: 
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(20,2.5))
    elif n_runs == 5:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, figsize=(20,2.5))
    elif n_runs == 7:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(nrows=1, ncols=7, sharey=True, figsize=(30,2.5))
    elif n_runs == 10:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, figsize=(28,2.5)) 
    elif n_runs == 15: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15) = plt.subplots(nrows=1, ncols=15, sharey=True, figsize=(20,2.5))
    elif n_runs == 9:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(nrows=1, ncols=9, sharey=True, figsize=(35,2.5)) 
    elif n_runs == 20: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(40,2.5))
    
    if idx==None: 
        count_players_with_title = long_titles.count(title)
        plt.suptitle(f'No of times each opponent was selected by {title} players (x{count_players_with_title}) (last 100) \n ', y=1.1, fontsize=18)
    else:
        print(f'plotting for idx {idx} only')
        plt.suptitle(f'No of times each opponent was selected by {title} player {idx} (last 100) \n ', y=1.1, fontsize=18)
        
    for run in range(1,n_runs+1):
        print(f'run {run}')
        run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)
    
        ax = locals()["ax"+str(run)]
        ax.set_title(f'\n run{run}') #No of times each action was played 

        if idx==None: #if plotting for allp layers with this title / type - NOTE need to fix opponents in this case! 
            small_run_df = run_df[run_df['title_p1']==title].iloc[-100:]
        else:
            small_run_df = run_df[(run_df['title_p1'] == title) & (run_df['idx_p1']==idx)].iloc[-100:]

        small_run_df['label_p2'] = small_run_df['title_p2'] + small_run_df['idx_p2'].astype(str)

        chart = sns.countplot(ax=ax, data=small_run_df, x=small_run_df['label_p2'], order=order, palette=palette)
        #sns.set(font_scale = 0.5)
        chart.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=90, 
        horizontalalignment='center',
        fontweight='light',
        fontsize='xx-small'
        )
        #chart.set_title(f'Selected by {title} players')
        
        #chart.bar_label(ax.containers[0])
        #for i in chart.containers:
        #    chart.bar_label(i,)
        for p in ax.patches:
            ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()+0.01), fontsize=10)

        chart.set(xlabel='')
    
    plt.xlabel('selection_player1') #to save space 

def plot_opponents_selected_aggregated_last100(destination_folder, n_runs, palette, order, title='Selfish'): #opponent_titles
    '''aggregated across all runs, plot the opponents selected by Player1==title'''

    results_df = pd.DataFrame(index=range(100000-100,100000))

    for run in range(1,n_runs+1):
        print(f'run {run}')
        run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0).iloc[-100:]
        small_run_df = run_df[run_df['title_p1']==title]
        results_df['run'+str(run)] = small_run_df['title_p2']
    
    plt.title(f'Opponents selected by {title} players \n'+f'(last 100) \n '+f'(aggregated across {n_runs} runs)') #No of times each action was played 

    data = pd.DataFrame({'selection':results_df.stack().reset_index(drop=True)})

    chart = sns.countplot(data=data, x=data['selection'], order=order, palette=palette) #sns.countplot(results_df.stack().reset_index(drop=True)) #x=results_df['title_p2'],
    #sns.set(font_scale = 0.5)
    chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='small'
    )
    #chart.set_title(f'Selected by {title} players')
    chart.set(xlabel='')
    
    plt.xlabel('selection_player1') #to save space 

def plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange, palette, order, title='Selfish'): #opponent_titles
    '''aggregated across all runs, plot the opponents selected by Player1==title'''

    if customrange: 
        limit1 = customrange[0]
        limit2 = customrange[1]
        print(f'customrange used, [{limit1}:{limit2}]')

    else: #if customrange not specified, use entire dataframe of results 
        limit1 = 0 
        limit2 = population_size*num_iter
    results_df = pd.DataFrame(index=range(limit1,limit2))


    for run in range(1,n_runs+1):
        #print(f'run {run}')
        run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0).iloc[limit1:limit2]
        small_run_df = run_df[run_df['title_p1']==title]
        results_df['run'+str(run)] = small_run_df['title_p2']
    
    plt.clf()
    plt.title(f'Opponent types selected by {title} players \n'+f'(Iterations {limit1}-{limit2}) \n '+f'(aggregated across {n_runs} runs)') #No of times each action was played 

    data = pd.DataFrame({'selection':results_df.stack().reset_index(drop=True)})

    chart = sns.countplot(data=data, x=data['selection'], order=order, palette=palette) #sns.countplot(results_df.stack().reset_index(drop=True)) #x=results_df['title_p2'],
    #sns.set(font_scale = 0.5)
    chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='small'
    )
    #chart.set_title(f'Selected by {title} players')
    chart.set(xlabel='')

    plt.show()
    
    #plt.xlabel('selection_player1') #to save space 


def plot_actions_player_old(destination_folder, n_runs, actions, title='Selfish'): 
    '''separately for each run, plot the actions played by Player1==title 
    NOTE this is a custom function for the piopulation S,AC,AD '''   
    if n_runs == 1: 
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True,figsize=(2.5,2.5))
    elif n_runs == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True,figsize=(7,2.5))
    elif n_runs == 5:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(11,2.5))
    elif n_runs == 9:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ) = plt.subplots(nrows=1, ncols=9, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 10:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 15: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15) = plt.subplots(nrows=1, ncols=15, sharey=True, sharex=True, figsize=(20,2.5))
    elif n_runs == 20: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(25,2.5))
    else: 
        print('number of runs not compatible')
        pass 
    plt.suptitle(f'No of times each action was played by {title} player (as selecting player) \n ', y=1.1, fontsize=22)
 
    for run in range(1,n_runs+1):
        print(f'run {run}')
        run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)

        #for plotting actions as both player1 and player2, use the following file:    
        #actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()

        ax = locals()["ax"+str(run)]
        ax.set_title(f'\n run{run}') #No of times each action was played 

        small_run_df = run_df[run_df['title_p1']==title]
        #small_run_df['action_player1'].replace({0:actions[0], 1:actions[1]}, inplace=True)
        small_run_df['action_player1'] = small_run_df['action_player1'].replace({0:actions[0], 1:actions[1]})

        #small_run_df.action_player1 = pd.Categorical(small_run_df.action_player1, categories=['C', 'D'], ordered=True)
        sns.countplot(ax=ax, data=small_run_df, x=small_run_df['action_player1'], order=actions) #col_wrap=3, , color=['green', 'orchid']

        ax.set(xlabel='')


        if False: 

            counts = pd.DataFrame(run_df[run_df['title_p1']==title]['action_player1'].value_counts())
            print(counts)
            counts.rename({0:'C', 1:'D'}, axis=0, inplace=True)
            counts.reset_index(inplace=True)
            counts.rename({'index':'Action', 'action_player1':'Count'}, axis=1, inplace=True)

            ax.bar(counts.loc[[0,1]]['index'], counts.loc[[0,1]]['action_player1'], color=['green', 'pink'], label=['C', 'D'])
            #counts.plot(kind='bar',label=['C', 'D'], color=['green', 'pink'], title=f'No of times each action was played \n run{run}', legend=False) 
    
    #fig.supxlabel('action_player1')

    plt.xlabel('action_player1') #to save space 

def plot_actions_player(destination_folder, n_runs, actions, title='Selfish', idx=0):
    '''separately for each run, plot the actions played by Player1==title 
    NOTE this is a custom function for the piopulation S,AC,AD '''   
    if n_runs == 1: 
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True,figsize=(2.5,2.5))
    elif n_runs == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True,figsize=(7,2.5))
    elif n_runs == 5:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(11,2.5))
    elif n_runs == 9:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ) = plt.subplots(nrows=1, ncols=9, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 10:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 15: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15) = plt.subplots(nrows=1, ncols=15, sharey=True, sharex=True, figsize=(20,2.5))
    elif n_runs == 20: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(25,2.5))
    else: 
        print('number of runs not compatible')
        pass 
    plt.suptitle(f'No of times each action was played by {title} player{idx} (as selecting player) \n ', y=1.1, fontsize=22)
 
    for run in range(1,n_runs+1):
        print(f'run {run}')
        
        run_df = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0)[f'run{run}']
        #for plotting actions as both player1 and player2, use the following file:    
        #actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()

        ax = locals()["ax"+str(run)]
        ax.set_title(f'\n run{run}') #No of times each action was played 

        small_run_df = run_df.sort_index(axis=0).dropna()

        #small_run_df.action_player1 = pd.Categorical(small_run_df.action_player1, categories=['C', 'D'], ordered=True)
        sns.countplot(ax=ax, data=small_run_df, x=small_run_df, order=actions) #col_wrap=3, , color=['green', 'orchid']

        ax.set(xlabel='')

    plt.xlabel('actions as player1 or player2') #to save space 



def plot_actions_player2(destination_folder, n_runs, actions, title='Selfish'): 
    '''separately for each run, plot the actions played by Player1==title 
    NOTE this is a custom function for the piopulation S,AC,AD '''   
    if n_runs == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True,figsize=(7,2.5))
    elif n_runs == 5:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(11,2.5))
    elif n_runs == 9:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ) = plt.subplots(nrows=1, ncols=9, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 10:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 15: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15) = plt.subplots(nrows=1, ncols=15, sharey=True, sharex=True, figsize=(20,2.5))
    elif n_runs == 20: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(25,2.5))
    else: 
        print('number of runs not compatible')
        pass 
    plt.suptitle(f'No of times each action was played by {title} player (as selectED player) \n ', y=1.1, fontsize=22)
 
    for run in range(1,n_runs+1):
        print(f'run {run}')
        run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)

        #for plotting actions as both player1 and player2, use the following file:    
        #actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()

        ax = locals()["ax"+str(run)]
        ax.set_title(f'\n run{run}') #No of times each action was played 

        small_run_df = run_df[run_df['title_p2']==title]
        #small_run_df['action_player1'].replace({0:actions[0], 1:actions[1]}, inplace=True)
        small_run_df['action_player2'] = small_run_df['action_player2'].replace({0:actions[0], 1:actions[1]})

        #small_run_df.action_player1 = pd.Categorical(small_run_df.action_player1, categories=['C', 'D'], ordered=True)
        sns.countplot(ax=ax, data=small_run_df, x=small_run_df['action_player2'], order=actions) #col_wrap=3, , color=['green', 'orchid']

        ax.set(xlabel='')

    #fig.supxlabel('action_player1')

    plt.xlabel('action_player2') #to save space 

def plot_actions_player_last100_old(destination_folder, n_runs, actions=['C', 'D'], title='Selfish'):
    '''separately for each run, plot the actions played by Player1==title 
    NOTE this is a custom function for the piopulation S,AC,AD '''   
    if n_runs == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True,figsize=(7,2.5))
    elif n_runs == 5:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(11,2.5))
    elif n_runs == 9:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ) = plt.subplots(nrows=1, ncols=9, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 10:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 15: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15) = plt.subplots(nrows=1, ncols=15, sharey=True, sharex=True, figsize=(20,2.5))
    elif n_runs == 20: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(25,2.5))
    else: 
        print('number of runs not compatible')
        pass 
    plt.suptitle(f'No of times each action was played by {title} player (as selecting player) (last 100) \n ', y=1.1, fontsize=22)
 
    for run in range(1,n_runs+1):
        print(f'run {run}')
        run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)
    
        ax = locals()["ax"+str(run)]
        ax.set_title(f'\n run{run}') #No of times each action was played 

        small_run_df1 = run_df[run_df['title_p1']==title]
        small_run_df2 = run_df[run_df['title_p2']==title]
        small_run_df = small_run_df1.append(small_run_df2).sort_index(axis=0).iloc[-100:]

        #small_run_df['action_player1'].replace({0:actions[0], 1:actions[1]}, inplace=True)
        small_run_df['action_player1'] = small_run_df['action_player1'].replace({0:actions[0], 1:actions[1]})
        small_run_df['action_player2'] = small_run_df['action_player2'].replace({0:actions[0], 1:actions[1]})
#TO DO 
        #small_run_df.action_player1 = pd.Categorical(small_run_df.action_player1, categories=['C', 'D'], ordered=True)
        sns.countplot(ax=ax, data=small_run_df, x=small_run_df['action_player1'], order=actions) #col_wrap=3, , color=['green', 'orchid']

        ax.set(xlabel='')

    plt.xlabel('action_player1') #to save space

def plot_actions_player_last100(destination_folder, n_runs, actions=['C', 'D'], title='Selfish', idx=0): 
    '''separately for each run, plot the actions played by Player1==title AND Player2==title
    NOTE this is a custom function for the piopulation S,AC,AD '''   
    if n_runs == 1:
        fig, (ax1) =  plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True,figsize=(2.5,2.5))
    elif n_runs == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True,figsize=(7,2.5))
    elif n_runs == 5:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(11,2.5))
    elif n_runs == 9:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ) = plt.subplots(nrows=1, ncols=9, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 10:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(17,2.5)) 
    elif n_runs == 15: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15) = plt.subplots(nrows=1, ncols=15, sharey=True, sharex=True, figsize=(20,2.5))
    elif n_runs == 20: 
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(25,2.5))
    else: 
        print('number of runs not compatible')
        pass 
    plt.suptitle(f'No of times each action was played by {title} player{idx} (as selecting player) (last 100) \n ', y=1.1, fontsize=22)

    for run in range(1,n_runs+1):
        print(f'run {run}')
        run_df = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0)[f'run{run}']
    
        ax = locals()["ax"+str(run)]
        ax.set_title(f'\n run{run}') #No of times each action was played 

        small_run_df = run_df.sort_index(axis=0).dropna().iloc[-100:]

        #small_run_df.action_player1 = pd.Categorical(small_run_df.action_player1, categories=['C', 'D'], ordered=True)
        sns.countplot(ax=ax, data=small_run_df, x=small_run_df, order=actions) #col_wrap=3, , color=['green', 'orchid']

        ax.set(xlabel='')

    plt.xlabel('actions as player1 or payer2') #to save space


def plot_all_actions_player(destination_folder, run_idx, player_idx, actions=['C','D'], title='Selfish', color='black', episodes=True): 
    '''separately for each run, plot the actions played by player {title},_{player_idx} as playe1 or player2  
    NOTE this is a custom function for the piopulation S,AC,AD '''   
    
    print(f'run {run_idx}')
    actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{player_idx}.csv',index_col=0).sort_index()[f'run{run_idx}']
    if episodes: 
        actions_idx = actions_idx[actions_idx.notna()]
        midpoint = len(actions_idx)//2
    else: 
        midpoint = 50000
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=False,figsize=(4.5,3))
    sns.countplot(ax=ax1, data=actions_idx[:midpoint], x=actions_idx[:midpoint], order=actions) #col_wrap=3, , color=['green', 'orchid']
    ax1.set(xlabel='first half')
    ax1.xaxis.label.set_color(color)
    sns.countplot(ax=ax2, data=actions_idx[midpoint:], x=actions_idx[midpoint:], order=actions) #col_wrap=3, , color=['green', 'orchid']
    ax2.set(xlabel='second half', ylabel='')
    ax2.xaxis.label.set_color(color)
    plt.title(f'Actions by {title} player {player_idx}, run {run_idx}', x=0) #, y=1.1, fontsize=18


def plot_actions_player_episoderange(destination_folder, run_idx, player_idx, actions=['C','D'], title='Selfish', color='black'): 
    '''separately for each run, plot the actions played by player {title},_{player_idx} as playe1 or player2  
    NOTE this is a custom function for the piopulation S,AC,AD '''   
    
    print(f'run {run_idx}')
    actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{player_idx}.csv',index_col=0).sort_index()[f'run{run_idx}']

    #actions_idx = actions_idx[actions_idx.notna()]
    #episode_idx1 = episode_range[0]*population_size
    #episode_idx2 = episode_range[1]*population_size
    #actions_idx = actions_idx.iloc[episode_idx1:episode_idx2]
    print('assuming speicif episoderange (hard-coded in the function)')

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=1, ncols=6, sharey=True, sharex=False,figsize=(10,2))

    sns.countplot(ax=ax1, data=actions_idx[0:100*population_size], x=actions_idx[0:100*population_size], order=actions) #col_wrap=3, , color=['green', 'orchid']
    ax1.set(xlabel='Ep \n 0-\n100')
    ax1.xaxis.label.set_color(color)

    sns.countplot(ax=ax2, data=actions_idx[5000*population_size:5100*population_size], x=actions_idx[5000*population_size:5100*population_size], order=actions) #col_wrap=3, , color=['green', 'orchid']
    ax2.set(xlabel='Ep \n 5000-\n5100', ylabel='')
    ax2.xaxis.label.set_color(color)

    sns.countplot(ax=ax3, data=actions_idx[7500*population_size:7600*population_size], x=actions_idx[7500*population_size:7600*population_size], order=actions) #col_wrap=3, , color=['green', 'orchid']
    ax3.set(xlabel='Ep \n 7500-\n7600', ylabel='')
    ax3.xaxis.label.set_color(color)

    sns.countplot(ax=ax4, data=actions_idx[9000*population_size:9100*population_size], x=actions_idx[9000*population_size:9100*population_size], order=actions) #col_wrap=3, , color=['green', 'orchid']
    ax4.set(xlabel='Ep \n 9000-\n5100', ylabel='')
    ax4.xaxis.label.set_color(color)

    sns.countplot(ax=ax5, data=actions_idx[12000*population_size:12100*population_size], x=actions_idx[12000*population_size:12100*population_size], order=actions) #col_wrap=3, , color=['green', 'orchid']
    ax5.set(xlabel='Ep \n 12000-\n12100', ylabel='')
    ax5.xaxis.label.set_color(color)

    sns.countplot(ax=ax6, data=actions_idx[29900*population_size:30000*population_size], x=actions_idx[29900*population_size:30000*population_size], order=actions) #col_wrap=3, , color=['green', 'orchid']
    ax6.set(xlabel='Ep \n29900-\n30000', ylabel='')
    ax6.xaxis.label.set_color(color)

    plt.title(f'Actions by {title} player {player_idx}, run {run_idx}', x=-2.2) #, y=1.1, fontsize=18


def plot_all_selections_allplayers(destination_folder, run_idx, player_idx):
    run_df = pd.read_csv(f'history/run{run_idx}.csv', index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
    #drop NAs from the df - i.e. drop rows where a selection was not actually made 
    run_df_clean = run_df[run_df['selected_prev_move'].notna()]

    runname = 'run'+str(run_idx)
    data_selections = pd.DataFrame(columns=runname)
    data_availables = pd.DataFrame(columns=runname)

    try:
        data_selections[runname] = run_df_clean['selected_prev_move']==0
        data_availables[runname] = run_df_clean['cooperative_sel_available']==True
    except:
        print('failed at run :', runname)  

    data_selections['sum_across_runs'] = data_selections.sum(axis=1)
    data_availables['sum_across_runs'] = data_availables.sum(axis=1)

    result = data_selections['sum_across_runs']/data_availables['sum_across_runs'] * 100
    result_str = []
    for i in result: 
        if i == float(100):
            result_str.append('Coop.')
        elif i == float(0):
            result_str.append('Defector')
    result_series = pd.Series(result_str, dtype=object)

    midpoint = 200000
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=False,figsize=(4.5,3))
    sns.countplot(ax=ax1, data=result_series[:midpoint], x=result_series[:midpoint], order=['Coop.', 'Defector']) #col_wrap=3, , color=['green', 'orchid']
    ax1.set(xlabel='first half') #, xticks={float(100):'Cooperator', float(0):'Not Coop.'}
    #ax1.xaxis.label.set_color('black')
    sns.countplot(ax=ax2, data=result_series[midpoint:], x=result_series[midpoint:], order=['Coop.', 'Defector']) #col_wrap=3, , color=['green', 'orchid']
    ax2.set(xlabel='second half', ylabel='')
    #ax2.xaxis.label.set_color('black')
    plt.title(f'Selections by all players combined, run {run_idx}', x=0) #, y=1.1, fontsize=18


#### new functions for population ####

def read_new_Q_values_format(destination_folder, player_idx, run_idx, Qtype):
    q = pd.read_pickle(f'{destination_folder}/QVALUES/Q_VALUES_{Qtype}_local_player{player_idx}_list_run{run_idx}.pickle')
    df = pd.Series(q).reset_index(name=f'Qvalue_{Qtype}')
    #df.columns=df.iloc[0] 
    #df.drop(0, axis=0, inplace=True)
    df.columns = ['state, action, iteration', f'Q-value {Qtype}']
    df[['state', 'action', 'player_tstep']] = pd.DataFrame(df['state, action, iteration'].apply(lambda x: make_tuple(x)).tolist(), index=df.index)
    df.drop(['state, action, iteration'], axis=1, inplace=True)

    df['state'] = df['state'].astype(str)
    df['action'] = df['action'].astype(int)
    df['player_tstep'] = df['player_tstep'].astype(int)
    df[f'Q-value {Qtype}'] = df[f'Q-value {Qtype}'].astype(float)
    
    return df#['Q-value dilemma']

def plot_one_run_dilemma_Q_values_population_statenotpairs(Q_values_df, run_idx, player_idx, maxq, linestyle='-', marker=','):
    '''plot progression of Q-value updates over the 10000 iterations for one example run - separately for each state.
    We plot both actions (C and D) on one plot to compare which action ends up being optimal in every state.'''
    Qtype='dilemma'
    gamma = 0.99 #set in config within each experiment

    rXs0a0_list = Q_values_df[(Q_values_df['state']==str(0)) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
    rXs0a1_list = Q_values_df[(Q_values_df['state']==str(0)) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
    rXs1a0_list = Q_values_df[(Q_values_df['state']==str(1)) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
    rXs1a1_list = Q_values_df[(Q_values_df['state']==str(1)) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
    
    #for debugging
    #rXs0a0_list = rXs0a0_list[(rXs0a0_list.index >= 100) & (rXs0a0_list.index < 160)]
    #rXs0a1_list = rXs0a1_list[(rXs0a1_list.index >= 100) & (rXs0a1_list.index < 160)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    #fig.suptitle(f'Q-values for all state-action pairs, run{run_idx} \n')
    #fig.suptitle(str(f'run{run_idx}, player{player_idx}'+'\n'+'\n'))
    ax1.plot(rXs0a0_list.index, rXs0a0_list, linestyle=linestyle, marker=marker, label=f'Action=C', color='darkgreen')
    ax1.plot(rXs0a1_list.index, rXs0a1_list, linestyle=linestyle, marker=marker, label=f'Action=D', color='purple')
#    ax1.plot(range(len(rXs0a0_list)), rXs0a0_list[:], label=f'Action=C', color='darkgreen')
#    ax1.plot(range(len(rXs0a1_list)), rXs0a1_list[:], label=f'Action=D', color='purple')
    ax1.set_title('State = C')
    ax2.plot(rXs1a0_list.index, rXs1a0_list, linestyle=linestyle, marker=marker, label=f'Action=C', color='darkgreen')
    ax2.plot(rXs1a1_list.index, rXs1a1_list, linestyle=linestyle, marker=marker, label=f'Action=D', color='purple')
    ax2.set_title('State = D')

    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    ax1.set(ylabel='Q-value')
    plt.suptitle(str(f'run{run_idx}, player{player_idx}'+'\n'), y=1.1)

    #max_Q_value = max(max(rXs0a0_list), max(rXs0a1_list), max(rXs1a0_list), max(rXs1a1_list)) #max q-value on this run
    #plt.ylim(0, int(max_Q_value+5))
    #set y axis limit based on the claculation of a max possible return in this game and the discount factor: r=4, gamma=0.9, max return = r*(1/(1-gamma_)
    if game_title == 'IPD': 
        plt.ylim(0, maxq*(gamma/(1-gamma))+5) #20
    #elif game_title == 'VOLUNTEER':
    #    plt.ylim(0, 50+5)
    #elif game_title == 'STAGHUNT':
    #    plt.ylim(0, 50+5)
    #then export the interactive outupt / single plot as pdf/html to store the results compactly 


def plot_one_run_dilemma_diff_Q_values_population_statenotpairs(Q_values_df, run_idx, player_idx):
    '''plot progression of Q-value updates over the 100000 iterations for one example run - separately for each state.
    We plot both actions (C and D) on one plot to compare which action ends up being optimal in every state.'''
    Qtype='dilemma'
    #force-set index to full number of iterations
    rXs0a0 = pd.Series(index=range(len(Q_values_df)), dtype='str') # [None] * num_iter #num_iter defined globally 
    rXs0a0_list = Q_values_df[(Q_values_df['state']==str(0)) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
    rXs0a0.update(rXs0a0_list)
    rXs0a1 = pd.Series(index=range(len(Q_values_df)), dtype='str') # [None] * num_iter #num_iter defined globally 
    rXs0a1_list = Q_values_df[(Q_values_df['state']==str(0)) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
    rXs0a1.update(rXs0a1_list)

    #forward-fill from past values 
    rXs0a0.fillna(method='ffill', inplace=True)
    rXs0a1.fillna(method='ffill', inplace=True)

    #calcullate difference between Q(C)-Q(D)
    rXs0 = rXs0a0-rXs0a1

    #repeat for state 1 
    rXs1a0 = pd.Series(index=range(len(Q_values_df)), dtype='str') 
    rXs1a0_list = Q_values_df[(Q_values_df['state']==str(1)) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
    rXs1a0.update(rXs1a0_list)
    rXs1a1 = pd.Series(index=range(len(Q_values_df)), dtype='str') 
    rXs1a1_list = Q_values_df[(Q_values_df['state']==str(1)) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
    rXs1a1.update(rXs1a1_list)

    rXs1a0.fillna(method='ffill', inplace=True)
    rXs1a1.fillna(method='ffill', inplace=True)

    rXs1 = rXs1a0-rXs1a1

    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    #fig.suptitle(f'Q-values for all state-action pairs, run{run_idx} \n')
    #fig.suptitle(str(f'run{run_idx}, player{player_idx}'+'\n'+'\n'))
    ax1.plot(rXs0.index, rXs0[:], label=f'Difference Q(C)-Q(D)', color='darkred', linewidth=0.6)
    ax1.plot(rXs0.index, [0]*len(rXs0), color='grey')

    ax1.set_title('State = C')
    ax2.plot(rXs1.index, rXs1[:], label=f'Difference Q(C)-Q(D)', color='darkred', linewidth=0.6)
    ax2.plot(rXs1.index, [0]*len(rXs1), color='grey')
    ax2.set_title('State = D')

    leg = plt.legend() # get the legend object
    for line in leg.get_lines(): # change the line width for the legend
        line.set_linewidth(4.0)
    ax1.set(ylabel='Q-value')
    plt.suptitle(str(f'run{run_idx}, player{player_idx}'+'\n'), y=1.1)


def plot_one_run_selection_Q_values_population_statenotpairs(Q_values_df, run_idx, player_idx, opponents, colors=['blue','cyan'], 
                                                             linestyle='-', marker=','):
    '''plot progression of Q-value updates over the num_iter iterations for one example run - separately for each state.
    We plot al; selections (every player in population excelpt myself) on one plot to compare which selection ends up being optimal in every state.'''
    Qtype='selection'
    gamma = 0.99

    if 0 in Q_values_df.index:
        Q_values_df.drop(0, axis=0, inplace=True) #drop first row where the player reacts to a random state 

    if population_size == 2: 
        only_states = Q_values_df['state'][1:].value_counts().index
        #only_states should be of length 1 or 2
        if len(only_states) == 1:
            rXs0a0_list = Q_values_df[(Q_values_df['state']==only_states[0]) & (Q_values_df['action']==0)][f'Q-value {Qtype}']

            fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
            fig.suptitle(str(f'run{run_idx}, player{player_idx}'), x=1)

            opponent1 = opponents[0]
            if opponent1 == 'AC':
                color1 = 'green'
            elif opponent1 == 'AD':
                color1 = 'orange'
            else: 
                color1 = 'blue'

            ax1.plot(rXs0a0_list.index, rXs0a0_list[:], label=f'Selection={opponent1}', color=color1, alpha=0.7)
            ax1.set_title(f'State = {only_states[0].replace("0", "C").replace("1","D").replace(" ","")}')
            ax1.set(xlabel='t_step')


        elif len(only_states) == 2:  
            rXs0a0_list = Q_values_df[(Q_values_df['state']==only_states[0]) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
            rXs0a1_list = Q_values_df[(Q_values_df['state']==only_states[0]) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
            rXs1a0_list = Q_values_df[(Q_values_df['state']==only_states[1]) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
            rXs1a1_list = Q_values_df[(Q_values_df['state']==only_states[1]) & (Q_values_df['action']==1)][f'Q-value {Qtype}'] 

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            fig.suptitle(str(f'run{run_idx}, player{player_idx}'), x=1)

            opponent1 = opponents[0]
            if opponent1 == 'AC':
                color1 = 'green'
            elif opponent1 == 'AD':
                color1 = 'orange'
            else: 
                color1 = 'blue'

            ax1.plot(rXs0a0_list.index, rXs0a0_list[:], label=f'Selection={opponent1}', color=color1, alpha=0.7)
            ax1.set_title(f'State = {only_states[0].replace("0", "C").replace("1","D").replace(" ","")}')
            ax1.set(xlabel='t_step')
            ax2.plot(rXs1a0_list.index, rXs1a0_list[:], label=f'Selection={opponent1}', color=color1, alpha=0.9)
            ax2.set_title(f'State = {only_states[1].replace("0", "C").replace("1","D").replace(" ","")}')
            ax2.set(xlabel='t_step')

        leg = plt.legend(loc='upper center', bbox_to_anchor=(-0.05, -0.3),
            ncol=5, fancybox=True, shadow=True) # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        ax1.set(ylabel='Q-value')
        plt.suptitle(str(f'run{run_idx}, player{player_idx}, all states'+'\n'), y=1.1)
        plt.show()
    
    elif population_size == 3: 
        rXs0a0_list = Q_values_df[(Q_values_df['state']==str((0,0))) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
        rXs0a1_list = Q_values_df[(Q_values_df['state']==str((0,0))) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
        rXs1a0_list = Q_values_df[(Q_values_df['state']==str((0,1))) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
        rXs1a1_list = Q_values_df[(Q_values_df['state']==str((0,1))) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
        rXs2a0_list = Q_values_df[(Q_values_df['state']==str((1,0))) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
        rXs2a1_list = Q_values_df[(Q_values_df['state']==str((1,0))) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
        rXs3a0_list = Q_values_df[(Q_values_df['state']==str((1,1))) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
        rXs3a1_list = Q_values_df[(Q_values_df['state']==str((1,1))) & (Q_values_df['action']==1)][f'Q-value {Qtype}']    

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
        fig.suptitle(str(f'run{run_idx}, player{player_idx}'), x=1)

        opponent1 = opponents[0]
        opponent2 = opponents[1]
        if opponent1 == 'AC':
            color1 = 'green'
        else: 
            color1 = colors[0]
        
        if opponent2 == 'AD':
            color2 = 'orange'
        else: 
            color2 = colors[1]
        #opponent3, color3 = 'S3', 'green'
        #opponent4, color4 = 'S4', 'blue'
        #opponent5, color5 = 'S5', 'red'
        #cyan, purple

        ax1.plot(rXs0a0_list.index, rXs0a0_list[:], label=f'Selection={opponent1}', color=color1, alpha=0.7)
        ax1.plot(rXs0a1_list.index, rXs0a1_list[:], label=f'Selection={opponent2}', color=color2, alpha=0.7)
        ax1.set_title('State = (C,C)')
        ax1.set(xlabel='t_step')
        ax2.plot(rXs1a0_list.index, rXs1a0_list[:], label=f'Selection={opponent1}', color=color1, alpha=0.9)
        ax2.plot(rXs1a1_list.index, rXs1a1_list[:], label=f'Selection={opponent2}', color=color2, alpha=0.9)
        ax2.set_title('State = (C,D)')
        ax2.set(xlabel='t_step')
        ax3.plot(rXs2a0_list.index, rXs2a0_list[:], label=f'Selection={opponent1}', color=color1, alpha=0.9)
        ax3.plot(rXs2a1_list.index, rXs2a1_list[:], label=f'Selection={opponent2}', color=color2, alpha=0.9)
        ax3.set_title('State = (D,C)')
        ax3.set(xlabel='t_step')
        ax4.plot(rXs3a0_list.index, rXs3a0_list[:], label=f'Selection={opponent1}', color=color1, alpha=0.9)
        ax4.plot(rXs3a1_list.index, rXs3a1_list[:], label=f'Selection={opponent2}', color=color2, alpha=0.9)
        ax4.set_title('State = (D,D)')
        ax4.set(xlabel='t_step')

        leg = plt.legend(loc='upper center', bbox_to_anchor=(-1.4, -0.3),
            ncol=5, fancybox=True, shadow=True) # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        ax1.set(ylabel='Q-value')
        plt.suptitle(str(f'run{run_idx}, player{player_idx}, all states'+'\n'), y=1.1)
        plt.show()
    
    elif population_size == 4: 
        common_states = Q_values_df['state'].value_counts()[0:4].index

        if len(common_states) == 1: 
            fix, (ax1) = plt.subplots(1, 1, figsize=(5,4), sharey=True)
        elif len(common_states) == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        elif len(common_states) == 3:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 4), sharey=True)
        else: 
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 4), sharey=True)
        #fig.suptitle(str(f'run{run_idx}, player{player_idx}, first 4 states'), x=1)
        colors = ['purple', 'blue', 'cyan', 'green']#, 'red'] #'purple', 'brown', 'pink', 'gray', 'olive']
        
        ax_counter = 0
        for state in common_states: 
            ax_counter += 1
            rXs0a0_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
            rXs0a1_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
            rXs0a2_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==2)][f'Q-value {Qtype}']

            ax = locals()["ax"+str(ax_counter)]
            #NOTE manual color assignment below, assuming we plot selections for player0 
            ax.plot(rXs0a0_list.index, rXs0a0_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[0]}', color=colors[1], alpha=0.7)
            ax.plot(rXs0a1_list.index, rXs0a1_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[1]}', color=colors[2], alpha=0.7)
            ax.plot(rXs0a2_list.index, rXs0a2_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[2]}', color=colors[3], alpha=0.7)

            ax.set_title(f'State = {state.replace("0", "C").replace("1","D").replace(" ","")}')
            ax.set(xlabel='t_step')

        leg = plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.3),
            ncol=2, fancybox=True, shadow=True) # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        ax1.set(ylabel='Q-value')
        plt.suptitle(str(f'run{run_idx}, player{player_idx}, most common 4 states'+'\n'), y=1.1)
        plt.show()
  
    elif population_size == 5: 
        common_states = Q_values_df['state'].value_counts()[0:4].index

        if len(common_states) == 1: 
            fix, (ax1) = plt.subplots(1, 1, figsize=(5,4), sharey=True)
        if len(common_states) == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        elif len(common_states) == 3:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 4), sharey=True)
        else: 
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 4), sharey=True)
        #fig.suptitle(str(f'run{run_idx}, player{player_idx}, first 4 states'), x=1)
        colors = ['purple', 'blue', 'cyan', 'green', 'red'] #'purple', 'brown', 'pink', 'gray', 'olive']
        
        ax_counter = 0
        for state in common_states: 
            ax_counter += 1
            rXs0a0_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
            rXs0a1_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
            rXs0a2_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==2)][f'Q-value {Qtype}']
            rXs0a3_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==3)][f'Q-value {Qtype}']

            ax = locals()["ax"+str(ax_counter)]
            #NOTE manual color assignment below, assuming we plot selections for player0 
            ax.plot(rXs0a0_list.index, rXs0a0_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[0]}', color=colors[1], alpha=0.7)
            ax.plot(rXs0a1_list.index, rXs0a1_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[1]}', color=colors[2], alpha=0.7)
            ax.plot(rXs0a2_list.index, rXs0a2_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[2]}', color=colors[3], alpha=0.7)
            ax.plot(rXs0a3_list.index, rXs0a3_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[3]}', color=colors[4], alpha=0.7)

            ax.set_title(f'State = {state.replace("0", "C").replace("1","D").replace(" ","")}')
            ax.set(xlabel='t_step')

        leg = plt.legend(loc='upper center', bbox_to_anchor=(-0.4, -0.3),
            ncol=2, fancybox=True, shadow=True) # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        ax1.set(ylabel='Q-value')
        plt.suptitle(str(f'run{run_idx}, player{player_idx}, most common 4 states'+'\n'), y=1.1)
        plt.show()
   
    elif population_size == 6: 
        common_states = Q_values_df['state'].value_counts()[0:4].index

        if len(common_states) == 1: 
            fix, (ax1) = plt.subplots(1, 1, figsize=(5,4), sharey=True)
        if len(common_states) == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        elif len(common_states) == 3:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 4), sharey=True)
        else: 
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 4), sharey=True)
        #fig.suptitle(str(f'run{run_idx}, player{player_idx}, first 4 states'), x=1)
        colors = ['purple', 'blue', 'cyan', 'green', 'red'] #'purple', 'brown', 'pink', 'gray', 'olive']
        
        ax_counter = 0
        for state in common_states: 
            ax_counter += 1
            rXs0a0_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
            rXs0a1_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
            rXs0a2_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==2)][f'Q-value {Qtype}']
            rXs0a3_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==3)][f'Q-value {Qtype}']

            ax = locals()["ax"+str(ax_counter)]
            #NOTE manual color assignment below, assuming we plot selections for player0 
            ax.plot(rXs0a0_list.index, rXs0a0_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[0]}', color=colors[1], alpha=0.7)
            ax.plot(rXs0a1_list.index, rXs0a1_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[1]}', color=colors[2], alpha=0.7)
            ax.plot(rXs0a2_list.index, rXs0a2_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[2]}', color=colors[3], alpha=0.7)
            ax.plot(rXs0a3_list.index, rXs0a3_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[3]}', color=colors[4], alpha=0.7)

            ax.set_title(f'State = {state.replace("0", "C").replace("1","D").replace(" ","")}')
            ax.set(xlabel='t_step')

        leg = plt.legend(loc='upper center', bbox_to_anchor=(-0.4, -0.3),
            ncol=2, fancybox=True, shadow=True) # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        ax1.set(ylabel='Q-value')
        plt.suptitle(str(f'run{run_idx}, player{player_idx}, most common 4 states'+'\n'), y=1.1)
        plt.show()

    elif population_size == 10: 
        common_states = Q_values_df['state'].value_counts()[0:4].index

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 4), sharey=True)
        #fig.suptitle(str(f'run{run_idx}, player{player_idx}, first 4 states'), x=1)
        #colors = ['blue', 'cyan', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        ax_counter = 0
        for state in common_states: 
            ax_counter += 1
            rXs0a0_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
            rXs0a1_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
            rXs0a2_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==2)][f'Q-value {Qtype}']
            rXs0a3_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==3)][f'Q-value {Qtype}']
            rXs0a4_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==4)][f'Q-value {Qtype}']
            rXs0a5_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==5)][f'Q-value {Qtype}']
            rXs0a6_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==6)][f'Q-value {Qtype}']
            rXs0a7_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==7)][f'Q-value {Qtype}']
            rXs0a8_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==8)][f'Q-value {Qtype}']

            ax = locals()["ax"+str(ax_counter)]

            ax.plot(rXs0a0_list.index, rXs0a0_list[:], linestyle=linestyle, marker=marker,label=f'Selection={opponents[0]}', alpha=0.7)
            ax.plot(rXs0a1_list.index, rXs0a1_list[:], linestyle=linestyle, marker=marker,label=f'Selection={opponents[1]}', alpha=0.7)
            ax.plot(rXs0a2_list.index, rXs0a2_list[:], linestyle=linestyle, marker=marker,label=f'Selection={opponents[2]}', alpha=0.7)
            ax.plot(rXs0a3_list.index, rXs0a3_list[:], linestyle=linestyle, marker=marker,label=f'Selection={opponents[3]}', alpha=0.7)
            ax.plot(rXs0a4_list.index, rXs0a4_list[:], linestyle=linestyle, marker=marker,label=f'Selection={opponents[4]}', alpha=0.7)
            ax.plot(rXs0a5_list.index, rXs0a5_list[:], linestyle=linestyle, marker=marker,label=f'Selection={opponents[5]}', alpha=0.7)
            ax.plot(rXs0a6_list.index, rXs0a6_list[:], linestyle=linestyle, marker=marker,label=f'Selection={opponents[6]}', alpha=0.7)
            ax.plot(rXs0a7_list.index, rXs0a7_list[:], linestyle=linestyle, marker=marker,label=f'Selection={opponents[7]}', alpha=0.7)
            ax.plot(rXs0a8_list.index, rXs0a8_list[:], linestyle=linestyle, marker=marker,label=f'Selection={opponents[8]}', alpha=0.7)
            ax.set_title(f'State = {state.replace("0", "C").replace("1","D").replace(" ","")}')
            ax.set(xlabel='t_step')

        leg = plt.legend(loc='upper center', bbox_to_anchor=(-1.4, -0.3),
            ncol=5, fancybox=True, shadow=True) # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        ax1.set(ylabel='Q-value')
        plt.suptitle(str(f'run{run_idx}, player{player_idx}, most common 4 states'+'\n'), y=1.1)
        plt.show()

    elif population_size == 11: 
        common_states = Q_values_df['state'].value_counts()[0:1].index

        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 4), sharey=True)
        #fig.suptitle(str(f'run{run_idx}, player{player_idx}, first 4 states'), x=1)
        #colors = ['blue', 'cyan', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        ax_counter = 0
        for state in common_states: 
            ax_counter += 1
            rXs0a0_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
            rXs0a1_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
            rXs0a2_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==2)][f'Q-value {Qtype}']
            rXs0a3_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==3)][f'Q-value {Qtype}']
            rXs0a4_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==4)][f'Q-value {Qtype}']
            rXs0a5_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==5)][f'Q-value {Qtype}']
            rXs0a6_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==6)][f'Q-value {Qtype}']
            rXs0a7_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==7)][f'Q-value {Qtype}']
            rXs0a8_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==8)][f'Q-value {Qtype}']
            rXs0a9_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==9)][f'Q-value {Qtype}']

            ax = locals()["ax"+str(ax_counter)]

            ax.plot(rXs0a0_list.index, rXs0a0_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[0]}', alpha=0.7)
            ax.plot(rXs0a1_list.index, rXs0a1_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[1]}', alpha=0.7)
            ax.plot(rXs0a2_list.index, rXs0a2_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[2]}', alpha=0.7)
            ax.plot(rXs0a3_list.index, rXs0a3_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[3]}', alpha=0.7)
            ax.plot(rXs0a4_list.index, rXs0a4_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[4]}', alpha=0.7)
            ax.plot(rXs0a5_list.index, rXs0a5_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[5]}', alpha=0.7)
            ax.plot(rXs0a6_list.index, rXs0a6_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[6]}', alpha=0.7)
            ax.plot(rXs0a7_list.index, rXs0a7_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[7]}', alpha=0.7)
            ax.plot(rXs0a8_list.index, rXs0a8_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[8]}', alpha=0.7)
            ax.plot(rXs0a9_list.index, rXs0a9_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[9]}', alpha=0.7)

            ax.set_title(f'State = {state.replace("0", "C").replace("1","D").replace(" ","")}')
            ax.set(xlabel='t_step')

        leg = plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.3),
            ncol=3, fancybox=True, shadow=True) # get the legend object
        #leg = plt.legend(loc='upper center', bbox_to_anchor=(-1.4, -0.3),
        #    ncol=5, fancybox=True, shadow=True) # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        ax1.set(ylabel='Q-value')
        plt.suptitle(str(f'run{run_idx}, player{player_idx}, most common 1 state'+'\n'), y=1.1)
        plt.show()

    elif population_size == 20: 
        common_states = Q_values_df['state'].value_counts()[0:4].index

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 4), sharey=True)
        #fig.suptitle(str(f'run{run_idx}, player{player_idx}, first 4 states'), x=1)
        #colors = ['blue', 'cyan', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        ax_counter = 0
        for state in common_states: 
            ax_counter += 1
            rXs0a0_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
            rXs0a1_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
            rXs0a2_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==2)][f'Q-value {Qtype}']
            rXs0a3_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==3)][f'Q-value {Qtype}']
            rXs0a4_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==4)][f'Q-value {Qtype}']
            rXs0a5_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==5)][f'Q-value {Qtype}']
            rXs0a6_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==6)][f'Q-value {Qtype}']
            rXs0a7_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==7)][f'Q-value {Qtype}']
            rXs0a8_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==8)][f'Q-value {Qtype}']
            rXs0a9_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==9)][f'Q-value {Qtype}']
            rXs0a10_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==10)][f'Q-value {Qtype}']
            rXs0a11_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==11)][f'Q-value {Qtype}']
            rXs0a12_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==12)][f'Q-value {Qtype}']
            rXs0a13_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==13)][f'Q-value {Qtype}']
            rXs0a14_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==14)][f'Q-value {Qtype}']
            rXs0a15_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==15)][f'Q-value {Qtype}']
            rXs0a16_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==16)][f'Q-value {Qtype}']
            rXs0a17_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==17)][f'Q-value {Qtype}']
            rXs0a18_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==18)][f'Q-value {Qtype}']


            ax = locals()["ax"+str(ax_counter)]

            ax.plot(rXs0a0_list.index, rXs0a0_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[0]}', alpha=0.7)
            ax.plot(rXs0a1_list.index, rXs0a1_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[1]}', alpha=0.7)
            ax.plot(rXs0a2_list.index, rXs0a2_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[2]}', alpha=0.7)
            ax.plot(rXs0a3_list.index, rXs0a3_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[3]}', alpha=0.7)
            ax.plot(rXs0a4_list.index, rXs0a4_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[4]}', alpha=0.7)
            ax.plot(rXs0a5_list.index, rXs0a5_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[5]}', alpha=0.7)
            ax.plot(rXs0a6_list.index, rXs0a6_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[6]}', alpha=0.7)
            ax.plot(rXs0a7_list.index, rXs0a7_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[7]}', alpha=0.7)
            ax.plot(rXs0a8_list.index, rXs0a8_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[8]}', alpha=0.7)
            ax.plot(rXs0a9_list.index, rXs0a9_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[9]}', alpha=0.7)
            ax.plot(rXs0a10_list.index, rXs0a10_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[10]}', alpha=0.7)
            ax.plot(rXs0a11_list.index, rXs0a11_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[11]}', alpha=0.7)
            ax.plot(rXs0a12_list.index, rXs0a12_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[12]}', alpha=0.7)
            ax.plot(rXs0a13_list.index, rXs0a13_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[13]}', alpha=0.7)
            ax.plot(rXs0a14_list.index, rXs0a14_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[14]}', alpha=0.7)
            ax.plot(rXs0a15_list.index, rXs0a15_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[15]}', alpha=0.7)
            ax.plot(rXs0a16_list.index, rXs0a16_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[16]}', alpha=0.7)
            ax.plot(rXs0a17_list.index, rXs0a17_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[17]}', alpha=0.7)
            ax.plot(rXs0a18_list.index, rXs0a18_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[18]}', alpha=0.7)

            ax.set_title(f'State = {state.replace("0", "C").replace("1","D").replace(" ","")}')
            ax.set(xlabel='t_step')

        leg = plt.legend(loc='upper center', bbox_to_anchor=(-1.4, -0.3),
            ncol=5, fancybox=True, shadow=True) # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        ax1.set(ylabel='Q-value')
        plt.suptitle(str(f'run{run_idx}, player{player_idx}, most common 4 states'+'\n'), y=1.1)
        plt.show()

    elif population_size == 21: 
        common_states = Q_values_df['state'].value_counts()[0:4].index

        #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 4), sharey=True)
        #fig.suptitle(str(f'run{run_idx}, player{player_idx}, first 4 states'), x=1)
        #colors = ['blue', 'cyan', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
        fig, (ax1) = plt.subplots(1, 1, figsize=(10,4), sharey=True)
        #if len(common_states) == 1: 
        #    fig, (ax1) = plt.subplots(1, 1, figsize=(5,4), sharey=True)
        #if len(common_states) == 2:
        #    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        #elif len(common_states) == 3:
        #    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 4), sharey=True)
        #else: 
        #    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 4), sharey=True)


        ax_counter = 0
        for state in common_states: 
            ax_counter += 1
            rXs0a0_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==0)][f'Q-value {Qtype}']
            rXs0a1_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==1)][f'Q-value {Qtype}']
            rXs0a2_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==2)][f'Q-value {Qtype}']
            rXs0a3_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==3)][f'Q-value {Qtype}']
            rXs0a4_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==4)][f'Q-value {Qtype}']
            rXs0a5_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==5)][f'Q-value {Qtype}']
            rXs0a6_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==6)][f'Q-value {Qtype}']
            rXs0a7_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==7)][f'Q-value {Qtype}']
            rXs0a8_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==8)][f'Q-value {Qtype}']
            rXs0a9_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==9)][f'Q-value {Qtype}']
            rXs0a10_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==10)][f'Q-value {Qtype}']
            rXs0a11_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==11)][f'Q-value {Qtype}']
            rXs0a12_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==12)][f'Q-value {Qtype}']
            rXs0a13_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==13)][f'Q-value {Qtype}']
            rXs0a14_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==14)][f'Q-value {Qtype}']
            rXs0a15_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==15)][f'Q-value {Qtype}']
            rXs0a16_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==16)][f'Q-value {Qtype}']
            rXs0a17_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==17)][f'Q-value {Qtype}']
            rXs0a18_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==18)][f'Q-value {Qtype}']
            rXs0a19_list = Q_values_df[(Q_values_df['state']==state) & (Q_values_df['action']==19)][f'Q-value {Qtype}']


            ax = locals()["ax"+str(ax_counter)]

            ax.plot(rXs0a0_list.index, rXs0a0_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[0]}', alpha=0.7)
            ax.plot(rXs0a1_list.index, rXs0a1_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[1]}', alpha=0.7)
            ax.plot(rXs0a2_list.index, rXs0a2_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[2]}', alpha=0.7)
            ax.plot(rXs0a3_list.index, rXs0a3_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[3]}', alpha=0.7)
            ax.plot(rXs0a4_list.index, rXs0a4_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[4]}', alpha=0.7)
            ax.plot(rXs0a5_list.index, rXs0a5_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[5]}', alpha=0.7)
            ax.plot(rXs0a6_list.index, rXs0a6_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[6]}', alpha=0.7)
            ax.plot(rXs0a7_list.index, rXs0a7_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[7]}', alpha=0.7)
            ax.plot(rXs0a8_list.index, rXs0a8_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[8]}', alpha=0.7)
            ax.plot(rXs0a9_list.index, rXs0a9_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[9]}', alpha=0.7)
            ax.plot(rXs0a10_list.index, rXs0a10_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[10]}', alpha=0.7)
            ax.plot(rXs0a11_list.index, rXs0a11_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[11]}', alpha=0.7)
            ax.plot(rXs0a12_list.index, rXs0a12_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[12]}', alpha=0.7)
            ax.plot(rXs0a13_list.index, rXs0a13_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[13]}', alpha=0.7)
            ax.plot(rXs0a14_list.index, rXs0a14_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[14]}', alpha=0.7)
            ax.plot(rXs0a15_list.index, rXs0a15_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[15]}', alpha=0.7)
            ax.plot(rXs0a16_list.index, rXs0a16_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[16]}', alpha=0.7)
            ax.plot(rXs0a17_list.index, rXs0a17_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[17]}', alpha=0.7)
            ax.plot(rXs0a18_list.index, rXs0a18_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[18]}', alpha=0.7)
            ax.plot(rXs0a19_list.index, rXs0a19_list[:], linestyle=linestyle, marker=marker, label=f'Selection={opponents[19]}', alpha=0.7)

            ax.set_title(f'State = {state.replace("0", "C").replace("1","D").replace(" ","")}')
            ax.set(xlabel='t_step')

        leg = plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.3),
            ncol=3, fancybox=True, shadow=True) # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        ax1.set(ylabel='Q-value')
        plt.suptitle(str(f'run{run_idx}, player{player_idx}, most common 4 states'+'\n'), y=1.1)
        plt.show()

    else: 
        print('function not defined for this poulation size')
    #set y axis limit based on the claculation of a max possible return in this game and the discount factor: r=4, gamma=0.9, max return = r*(1/(1-gamma_)
    #if game_title == 'IPD': 
    #    plt.ylim(0, maxq*(gamma/(1-gamma))+5) 
    #elif game_title == 'VOLUNTEER':
    #    plt.ylim(0, 50+5)
    #elif game_title == 'STAGHUNT':
    #    plt.ylim(0, 50+5)
    #then export the interactive outupt / single plot as pdf/html to store the results compactly 



def plot_Q_values_population_statenotpairs(destination_folder, population_size, n_runs, which, record_Qvalues, maxq, fewer_than_nruns = False): #NB set manually which Q values are being plotted
    if fewer_than_nruns == True:
        n_runs = 10 

    for player_idx in range(population_size):
            
        for run_idx in range(n_runs): 
            run_idx += 1 

            if which == 'local':
                #history_Qvalues_local_player = np.load(f'{destination_folder}/QVALUES/Q_VALUES_local_player{player_idx}_list_run{run_idx}.npy', allow_pickle=True)
                print(f'Printing local Q-values, player{player_idx}, run{run_idx}:')
                if record_Qvalues == 'both':
                    print('plotting DILEMMA Q values and diff')
                    history_Qvalues_DILEMMA_local_player = read_new_Q_values_format(destination_folder, player_idx, run_idx, Qtype = 'dilemma')
                    plot_one_run_dilemma_Q_values_population_statenotpairs(Q_values_df = history_Qvalues_DILEMMA_local_player, run_idx = run_idx, player_idx = player_idx, maxq = maxq)
                    plot_one_run_dilemma_diff_Q_values_population_statenotpairs(Q_values_df = history_Qvalues_DILEMMA_local_player, run_idx = run_idx, player_idx = player_idx)
                    
                    print('plotting SELECTION Q values and diff')
                    history_Qvalues_SELECTION_local_player = read_new_Q_values_format(destination_folder, player_idx, run_idx, Qtype = 'selection')
                    plot_one_run_selection_Q_values_population_statenotpairs(Q_values_df = history_Qvalues_SELECTION_local_player, run_idx = run_idx, player_idx = player_idx)
                    print('not implemented plotting differences yet')

                    #TODO 
                elif record_Qvalues == 'dilemma':
                    history_Qvalues_local_player = read_new_Q_values_format(destination_folder, player_idx, run_idx, Qtype = 'dilemma')
                    plot_one_run_dilemma_Q_values_population_statenotpairs(Q_values_df = history_Qvalues_local_player, run_idx = run_idx, player_idx = player_idx, maxq = maxq)
        #           plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_local_player, run_idx = run_idx)
                elif record_Qvalues == 'selection':                    
                    history_Qvalues_local_player = read_new_Q_values_format(destination_folder, player_idx, run_idx, Qtype = 'selection')
                    plot_one_run_selection_Q_values_population_statenotpairs(Q_values_df = history_Qvalues_local_player, run_idx = run_idx, player_idx = player_idx)


            elif which == 'target':
                pass 
                #try: 
                #    print(f'Printing target Q-values, player{player_idx}, , run{run_idx}:')
                #    history_Qvalues_target_player = np.load(f'{destination_folder}/QVALUES/Q_VALUES_target_{player_idx}_list_run{run_idx}.npy', allow_pickle=True)
                #    plot_one_run_Q_values_population(Q_values_list = history_Qvalues_target_player, run_idx = run_idx, player_idx = player_idx)
                #         plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_target_player, run_idx = run_idx)
                #except:
                #    print(f'Unable to locate target Q-values, player{player_idx}')



def read_new_Loss_format(destination_folder, player_idx, run_idx, LOSStype):
    '''read in Loss values per iteration for a single playeer & run, input pickle, output pd dataframe'''
    q = pd.read_pickle(f'{destination_folder}/LOSS/LOSS_{LOSStype}_player{player_idx}_list_run{run_idx}.pickle')
    df = pd.Series(q).reset_index(name=f'Loss_{LOSStype}')
    #df.columns=df.iloc[0] 
    #df.drop(0, axis=0, inplace=True)
    df.columns = ['player_tstep', f'Loss {LOSStype}']

    df['player_tstep'] = df['player_tstep'].astype(int)
    df[f'Loss {LOSStype}'] = df[f'Loss {LOSStype}'].astype(float)
    df.set_index('player_tstep', inplace=True)
    
    return df


def plot_Loss_population_newest_onerun(destination_folder, run_idx, population_size, LOSStype):

    if population_size <= 10: 
        fig, axs = plt.subplots(1, population_size, figsize=(10 * population_size, 15), sharey=True)
        #fig.subplots_adjust(hspace=0.5)
    elif population_size <= 20: 
        fig, axs = plt.subplots(2, int(population_size/2), figsize=(2 * population_size, 10), sharey=True)

    counter = -1 
    for player_idx in range(population_size):
        counter += 1 
        LOSS_player = read_new_Loss_format(destination_folder, player_idx, run_idx, LOSStype)
        
        if counter < 10:
            subplot_row = 0 
            subplot_col = player_idx
        else: 
            subplot_row = 1
            subplot_col = player_idx - 10
        ax = axs[subplot_row, subplot_col]
        ax.plot(LOSS_player, linewidth=0.7) #label='loss for DQN Player', 
        ax.set_xticklabels(ax.get_xticks(), rotation = 90)
        #ax.plot(loss_list[~np.isnan(loss_list)], label='loss for DQN Player', linewidth=0.1)
        #if min(loss_list):
        #    if max(loss_list):
        #        ax.set_ylim(min(loss_list) - 1, max(loss_list) + 1)
        #ax.set_ylim(min(loss_list[~np.isnan(loss_list)]) - 1, max(loss_list[~np.isnan(loss_list)]) + 1)

        ax.set_title(f'Player{player_idx}')

    plt.suptitle(f'{LOSStype} Loss over time, Run{run_idx}', y=1.1)
    plt.xlabel('Iteration')
    axs[0,0].set_ylabel('Loss')
    axs[1,0].set_ylabel('Loss')
    plt.tight_layout()
    plt.show()




if False: 
    ########################
    #### baseline group plots (learning vs. static opponent), or fuly static opponent ####
    ########################
    #TO DO UPDATE THESE
    def plot_basleline_relative_moral_reward(player1_title, n_runs):
        ##################################
        #### cumulative - game reward for player1_tytle vs others  ####
        ##################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
        against_AC_means = against_AC.mean(axis=1)
        against_AC_sds = against_AC.std(axis=1)
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
        against_AD_means = against_AD.mean(axis=1)
        against_AD_sds = against_AD.std(axis=1)
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
        against_TFT_means = against_TFT.mean(axis=1)
        against_TFT_sds = against_TFT.std(axis=1)
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_cumulative_reward_intrinsic.csv', index_col=0)
        against_Random_means = against_Random.mean(axis=1)
        against_Random_sds = against_Random.std(axis=1)
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(against_AC.index[:], against_AC_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AC')#, color='red')
        plt.fill_between(against_AC.index[:], against_AC_means-against_AC_ci, against_AC_means+against_AC_ci, alpha=0.5) #facecolor='#ff9999', 
        plt.plot(against_AD.index[:], against_AD_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AD')#, color='#556b2f')
        plt.fill_between(against_AD.index[:], against_AD_means-against_AD_ci, against_AD_means+against_AD_ci, alpha=0.5) #facecolor='#ccff99', 
        plt.plot(against_TFT.index[:], against_TFT_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_TFT')#, color='#00cccc')
        plt.fill_between(against_TFT.index[:], against_TFT_means-against_TFT_ci, against_TFT_means+against_TFT_ci, alpha=0.5) #facecolor='#99ffff', 
        plt.plot(against_Random.index[:], against_Random_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_Random')#, color='orange')
        plt.fill_between(against_Random.index[:], against_Random_means-against_Random_ci, against_Random_means+against_Random_ci, alpha=0.5) #facecolor='#ffcc99', 
        
        plt.title(r'Cumulative Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player (baseline)')
        if game_title=='IPD':
            if player1_title == 'QLUT':
                plt.gca().set_ylim([0, 60000])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-1300, 0]) 
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 10000])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 50000])
        elif game_title=='VOLUNTEER':
            if player1_title == 'QLUT':
                plt.gca().set_ylim([0, 80000])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-1300, 0]) 
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 10000])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 50000])
        elif game_title=='STAGHUNT': 
            if player1_title == 'QLUT':
                plt.gca().set_ylim([0, 80000])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-1300, 0]) 
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 10000])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 50000])
        plt.ylabel(f'Cumulative Intrinsoc reward for {player1_title}')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        if not os.path.isdir('results/outcome_plots/reward'):
            os.makedirs('results/outcome_plots/reward')
        plt.savefig(f'results/outcome_plots/reward/baseline_cumulative_intrinsic_reward_{player1_title}.png', bbox_inches='tight')


        ##################################
        #### non-cumulative - game reward for player1_title vs others  ####
        ##################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_reward_intrinsic.csv', index_col=0)
        against_AC_means = against_AC.mean(axis=1)
        against_AC_sds = against_AC.std(axis=1)
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_reward_intrinsic.csv', index_col=0)
        against_AD_means = against_AD.mean(axis=1)
        against_AD_sds = against_AD.std(axis=1)
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_reward_intrinsic.csv', index_col=0)
        against_TFT_means = against_TFT.mean(axis=1)
        against_TFT_sds = against_TFT.std(axis=1)
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_reward_intrinsic.csv', index_col=0)
        against_Random_means = against_Random.mean(axis=1)
        against_Random_sds = against_Random.std(axis=1)
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(against_AC.index[:], against_AC_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AC')#, color='red')
        plt.fill_between(against_AC.index[:], against_AC_means-against_AC_ci, against_AC_means+against_AC_ci, alpha=0.5) #facecolor='#ff9999', 
        plt.plot(against_AD.index[:], against_AD_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AD')#, color='#556b2f')
        plt.fill_between(against_AD.index[:], against_AD_means-against_AD_ci, against_AD_means+against_AD_ci, alpha=0.5) #facecolor='#ccff99', 
        plt.plot(against_TFT.index[:], against_TFT_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_TFT')#, color='#00cccc')
        plt.fill_between(against_TFT.index[:], against_TFT_means-against_TFT_ci, against_TFT_means+against_TFT_ci, alpha=0.5) #facecolor='#99ffff', 
        plt.plot(against_Random.index[:], against_Random_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_Random')#, color='orange')
        plt.fill_between(against_Random.index[:], against_Random_means-against_Random_ci, against_Random_means+against_Random_ci, alpha=0.5) #facecolor='#ffcc99', 
        
        plt.title(r'Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player (baseline)')

        if game_title=='IPD':
            if player1_title == 'QLUT':
                plt.gca().set_ylim([4, 6])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-5, 0]) 
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 1])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 5])
        elif game_title=='VOLUNTEER':
            if player1_title == 'QLUT':
                plt.gca().set_ylim([2, 8])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-5, 0]) 
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 1])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 5])
        elif game_title=='STAGHUNT': 
            if player1_title == 'QLUT':
                plt.gca().set_ylim([4, 10])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-5, 0]) 
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 1])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 5])

        plt.ylabel(f'Intrinsic reward for {player1_title}')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'results/outcome_plots/group_outcomes/baseline_intrinsic_reward_{player1_title}.png', bbox_inches='tight')

    def plot_baseline_cumulative_reward(player1_title, n_runs):
        #plot game and moral cumulative rewards as  barplots - how well off did the players end up relative to each other on the game / in terms of moral reward? 

        ##################################
        #### game cumulative reward for player1_tytle vs others  ####
        ##################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_AC_means = against_AC.mean()
        against_AC_sds = against_AC.std()
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_AD_means = against_AD.mean()
        against_AD_sds = against_AD.std()
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_TFT_means = against_TFT.mean()
        against_TFT_sds = against_TFT.std()
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_Random_means = against_Random.mean()
        against_Random_sds = against_Random.std()
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

    
        #import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.rcParams.update({'font.size':14})
        ax = fig.add_axes([0,0,1,1])
        labels = ['vs_AC', 'vs_AD', 'vs_TFT', 'vs_Random']
        means = [against_AC_means,against_AD_means,against_TFT_means,against_Random_means]
        cis = [against_AC_ci,against_AD_ci,against_TFT_ci,against_Random_ci]
        colors = ['cornflowerblue', 'orange', 'lightgreen', 'lightcoral']
        ax.bar(labels, means, yerr=cis, color=colors)
        ax.set_ylabel(f'Cumulative Game reward for {player1_title}')
        ax.set_title('Cumulative Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player')
        ax.set_xlabel('Opponent type')
        plt.savefig(f'results/outcome_plots/reward/bar_baseline_cumulative_game_reward_{player1_title}.png', bbox_inches='tight')


        ##################################
        #### moral cumulative reward for player1_tytle vs others  ####
        ##################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_AC_means = against_AC.mean()
        against_AC_sds = against_AC.std()
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_AD_means = against_AD.mean()
        against_AD_sds = against_AD.std()
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_TFT_means = against_TFT.mean()
        against_TFT_sds = against_TFT.std()
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_Random_means = against_Random.mean()
        against_Random_sds = against_Random.std()
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

    
        #import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.rcParams.update({'font.size':14})
        ax = fig.add_axes([0,0,1,1])
        labels = ['vs_AC', 'vs_AD', 'vs_TFT', 'vs_Random']
        means = [against_AC_means,against_AD_means,against_TFT_means,against_Random_means]
        cis = [against_AC_ci,against_AD_ci,against_TFT_ci,against_Random_ci]
        colors = ['cornflowerblue', 'orange', 'lightgreen', 'lightcoral']
        ax.bar(labels, means, yerr=cis, color=colors)
        ax.set_ylabel(f'Cumulative Intrinsic reward for {player1_title}')
        ax.set_title('Cumulative Intrinsic Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player')
        ax.set_xlabel('Opponent type')
        plt.savefig(f'results/outcome_plots/reward/bar_baseline_cumulative_intrinsic_reward_{player1_title}.png', bbox_inches='tight')

    def plot_basleline_relative_cooperation(player1_title, n_runs): 
        actions_against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/action.csv', index_col=0)
        #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
        actions_against_AC['%_defect'] = actions_against_AC[actions_against_AC[:]==1].count(axis='columns')
        actions_against_AC['%_cooperate'] = n_runs-actions_against_AC['%_defect']
        actions_against_AC['%_defect'] = (actions_against_AC['%_defect']/n_runs)*100
        actions_against_AC['%_cooperate'] = (actions_against_AC['%_cooperate']/n_runs)*100

        actions_against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/action.csv', index_col=0)
        actions_against_AD['%_defect'] = actions_against_AD[actions_against_AD[:]==1].count(axis='columns')
        actions_against_AD['%_cooperate'] = n_runs-actions_against_AD['%_defect']
        actions_against_AD['%_defect'] = (actions_against_AD['%_defect']/n_runs)*100
        actions_against_AD['%_cooperate'] = (actions_against_AD['%_cooperate']/n_runs)*100

        actions_against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/action.csv', index_col=0)
        actions_against_TFT['%_defect'] = actions_against_TFT[actions_against_TFT[:]==1].count(axis='columns')
        actions_against_TFT['%_cooperate'] = n_runs-actions_against_TFT['%_defect']
        actions_against_TFT['%_defect'] = (actions_against_TFT['%_defect']/n_runs)*100
        actions_against_TFT['%_cooperate'] = (actions_against_TFT['%_cooperate']/n_runs)*100

        actions_against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/action.csv', index_col=0)
        actions_against_Random['%_defect'] = actions_against_Random[actions_against_Random[:]==1].count(axis='columns')
        actions_against_Random['%_cooperate'] = n_runs-actions_against_Random['%_defect']
        actions_against_Random['%_defect'] = (actions_against_Random['%_defect']/n_runs)*100
        actions_against_Random['%_cooperate'] = (actions_against_Random['%_cooperate']/n_runs)*100

        #plot results 
        plt.figure(dpi=80) #figsize=(10, 6), 
        plt.plot(actions_against_AC.index[:], actions_against_AC['%_cooperate'], lw=0.5, alpha=0.5, label=f'{player1_title} vs. AC')#, color='blue')
        plt.plot(actions_against_AD.index[:], actions_against_AD['%_cooperate'], lw=0.5, alpha=0.5, label=f'{player1_title} vs. AD')#, color='blue')
        plt.plot(actions_against_TFT.index[:], actions_against_TFT['%_cooperate'], lw=0.5, alpha=0.5, label=f'{player1_title} vs. TFT')#, color='blue')
        plt.plot(actions_against_Random.index[:], actions_against_Random['%_cooperate'], lw=0.5, alpha=0.5, label=f'{player1_title} vs. Random')#, color='blue')
        
        plt.title('Probability of Cooperation (% over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player (baseline)')
        plt.gca().set_ylim([0, 100])
        plt.ylabel('Percentage cooperating')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        if not os.path.isdir('results/outcome_plots/cooperation'):
            os.makedirs('results/outcome_plots/cooperation')
        plt.savefig(f'results/outcome_plots/cooperation/baseline_cooperation_{player1_title}.png', bbox_inches='tight')

    def plot_relative_baseline_action_pairs(player1_title, n_runs):
        '''visualise action types that each individual player takes against their opponent's last move 
        --> what strategies are being learnt at all steps of the run? 
        - consider the whole run'''
        #NOTE this will only work for 10000 iterations right now, not fewer!!!
        #NOTE we plot after iteration 0 as then the agent is reacting to a default initial state, not a move from the opponent

        action_pairs_against_AC = pd.read_csv(f'results/{player1_title}_AC/action_pairs.csv', index_col=0)
        action_pairs_against_AD = pd.read_csv(f'results/{player1_title}_AD/action_pairs.csv', index_col=0)
        ction_pairs_against_TFT = pd.read_csv(f'results/{player1_title}_TFT/action_pairs.csv', index_col=0)
        action_pairs_against_Random = pd.read_csv(f'results/{player1_title}_Random/action_pairs.csv', index_col=0)

        #results_counts = action_pairs_against_QLS.transpose().apply(pd.value_counts).transpose()[1:] 
        #plot after first episode, as in the first apisode they are reacting to default state=0
        #results_counts.dropna(axis=1, how='all', inplace=True)

        action_pairs_against_AC['%_CC'] = action_pairs_against_AC[action_pairs_against_AC[:]=='C, C'].count(axis='columns')
        action_pairs_against_AD['%_CC'] = action_pairs_against_AD[action_pairs_against_AD[:]=='C, C'].count(axis='columns')
        ction_pairs_against_TFT['%_CC'] = ction_pairs_against_TFT[ction_pairs_against_TFT[:]=='C, C'].count(axis='columns')
        action_pairs_against_Random['%_CC'] = action_pairs_against_Random[action_pairs_against_Random[:]=='C, C'].count(axis='columns')


        #plt.figure(figsize=(20, 15), dpi=100)
        #results_counts.plot.area(stacked=True, ylabel = '# agent pairs taking this pair of actions \n (across '+str(n_runs)+' runs)', rot=45,
        #    xlabel='Iteration', #colormap='PiYG_r',
        #    color={'C, C':'#28641E', 'C, D':'#B0DC82', 'D, C':'#EEAED4', 'D, D':'#8E0B52'}, 
        #    title='Pairs of simultaneous actions over time: \n '+player1_title+' agent vs '+player2_title+' agent')
        #plt.savefig(f'{destination_folder}/plots/relative_action_pairs.png', bbox_inches='tight')

        #plot results 
        plt.figure(dpi=80) #figsize=(10, 6), 
        plt.plot(action_pairs_against_AC.index[:], action_pairs_against_AC['%_CC'], lw=0.5, alpha=0.5, label=f'{player1_title} vs. AC')#, color='blue')
        plt.plot(action_pairs_against_AD.index[:], action_pairs_against_AD['%_CC'], lw=0.5, alpha=0.5, label=f'{player1_title} vs. AD')#, color='blue')
        plt.plot(ction_pairs_against_TFT.index[:], ction_pairs_against_TFT['%_CC'], lw=0.5, alpha=0.5, label=f'{player1_title} vs. TFT')#, color='blue')
        plt.plot(action_pairs_against_Random.index[:], action_pairs_against_Random['%_CC'], lw=0.5, alpha=0.5, label=f'{player1_title} vs. Random')#, color='blue')

        plt.title('Mutual Cooperation among the two agents (% over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player (baseline)')
        plt.gca().set_ylim([0, 100])
        plt.ylabel('Percentage mutually cooperating')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        if not os.path.isdir('results/outcome_plots/cooperation'):
            os.makedirs('results/outcome_plots/cooperation')
        plt.savefig(f'results/outcome_plots/cooperation/baseline_mutual_cooperation_{player1_title}.png', bbox_inches='tight')

    def plot_baseline_relative_outcomes(type, player1_title, n_runs, game_title):
        ##################################
        #### cumulative - {type} game reward for player1_tytle vs others  ####
        ##################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/df_cumulative_reward_{type}.csv', index_col=0)
        against_AC_means = against_AC.mean(axis=1)
        against_AC_sds = against_AC.std(axis=1)
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/df_cumulative_reward_{type}.csv', index_col=0)
        against_AD_means = against_AD.mean(axis=1)
        against_AD_sds = against_AD.std(axis=1)
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/df_cumulative_reward_{type}.csv', index_col=0)
        against_TFT_means = against_TFT.mean(axis=1)
        against_TFT_sds = against_TFT.std(axis=1)
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/df_cumulative_reward_{type}.csv', index_col=0)
        against_Random_means = against_Random.mean(axis=1)
        against_Random_sds = against_Random.std(axis=1)
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(against_AC.index[:], against_AC_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AC')#, color='red')
        plt.fill_between(against_AC.index[:], against_AC_means-against_AC_ci, against_AC_means+against_AC_ci, alpha=0.5) #facecolor='#ff9999', 
        plt.plot(against_AD.index[:], against_AD_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AD')#, color='#556b2f')
        plt.fill_between(against_AD.index[:], against_AD_means-against_AD_ci, against_AD_means+against_AD_ci, alpha=0.5) #facecolor='#ccff99', 
        plt.plot(against_TFT.index[:], against_TFT_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_TFT')#, color='#00cccc')
        plt.fill_between(against_TFT.index[:], against_TFT_means-against_TFT_ci, against_TFT_means+against_TFT_ci, alpha=0.5) #facecolor='#99ffff', 
        plt.plot(against_Random.index[:], against_Random_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_Random')#, color='orange')
        plt.fill_between(against_Random.index[:], against_Random_means-against_Random_ci, against_Random_means+against_Random_ci, alpha=0.5) #facecolor='#ffcc99', 
        
        plt.title(r'Cumulative '+type+' Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player (baseline)')
        if game_title=='IPD':
            if type=='collective':
                plt.gca().set_ylim([0, 60000])
            elif type=='gini':
                plt.gca().set_ylim([0, 10000])
            elif type=='min':
                plt.gca().set_ylim([0, 30000])
        elif game_title=='VOLUNTEER':
            if type=='collective': 
                plt.gca().set_ylim([0, 80000])
            elif type=='gini':
                plt.gca().set_ylim([0, 10000])
            elif type=='min':
                plt.gca().set_ylim([0, 40000])
        elif game_title=='STAGHUNT': 
            if type=='collective': 
                plt.gca().set_ylim([0, 100000])
            elif type=='gini':
                plt.gca().set_ylim([0, 10000])
            elif type=='min':
                plt.gca().set_ylim([0, 50000])    
        plt.ylabel(f'Cumulative {type} reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        if not os.path.isdir('results/outcome_plots/group_outcomes'):
            os.makedirs('results/outcome_plots/group_outcomes')
        plt.savefig(f'results/outcome_plots/group_outcomes/baseline_cumulative_{type}_reward_{player1_title}.png', bbox_inches='tight')


        ######################################
        #### non-cumulative - {type} game reward ####
        ######################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/df_reward_{type}.csv', index_col=0)
        against_AC_means = against_AC.mean(axis=1)
        against_AC_sds = against_AC.std(axis=1)
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/df_reward_{type}.csv', index_col=0)
        against_AD_means = against_AD.mean(axis=1)
        against_AD_sds = against_AD.std(axis=1)
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/df_reward_{type}.csv', index_col=0)
        against_TFT_means = against_TFT.mean(axis=1)
        against_TFT_sds = against_TFT.std(axis=1)
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/df_reward_{type}.csv', index_col=0)
        against_Random_means = against_Random.mean(axis=1)
        against_Random_sds = against_Random.std(axis=1)
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(against_AC.index[:], against_AC_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_AC')#, color='red')
        plt.fill_between(against_AC.index[:], against_AC_means-against_AC_ci, against_AC_means+against_AC_ci, alpha=0.5) #facecolor='#ff9999', 
        plt.plot(against_AD.index[:], against_AD_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_AD')#, color='#556b2f')
        plt.fill_between(against_AD.index[:], against_AD_means-against_AD_ci, against_AD_means+against_AD_ci, alpha=0.5) #facecolor='#ccff99', 
        plt.plot(against_TFT.index[:], against_TFT_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_TFT')#, color='#00cccc')
        plt.fill_between(against_TFT.index[:], against_TFT_means-against_TFT_ci, against_TFT_means+against_TFT_ci, alpha=0.5) #facecolor='#99ffff', 
        plt.plot(against_Random.index[:], against_Random_means[:], lw=0.5, alpha=0.5, label=f'{player1_title}_Random')#, color='orange')
        plt.fill_between(against_Random.index[:], against_Random_means-against_Random_ci, against_Random_means+against_Random_ci, alpha=0.5) #facecolor='#ffcc99',
        
        plt.title(type+' Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player (baseline)')
        if type=='gini':
            plt.gca().set_ylim([0, 1])
        if game_title=='IPD':
            if type=='collective':
                plt.gca().set_ylim([4, 6])
            elif type=='min':
                plt.gca().set_ylim([1, 3])
        elif game_title=='VOLUNTEER':
            if type=='collective': 
                plt.gca().set_ylim([2, 8])
            elif type=='min':
                plt.gca().set_ylim([1, 4])
        elif game_title=='STAGHUNT': 
            if type=='collective': 
                plt.gca().set_ylim([4, 10])
            elif type=='min':
                plt.gca().set_ylim([1, 5])    
        plt.ylabel(f'{type} reward')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'results/outcome_plots/group_outcomes/baseline_{type}_reward_{player1_title}.png', bbox_inches='tight')

    def old_plot_basleline_relative_reward(player1_title, n_runs):
        ##################################
        #### cumulative - game reward for player1_tytle vs others  ####
        ##################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_cumulative_reward_game.csv', index_col=0)
        against_AC_means = against_AC.mean(axis=1)
        against_AC_sds = against_AC.std(axis=1)
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_cumulative_reward_game.csv', index_col=0)
        against_AD_means = against_AD.mean(axis=1)
        against_AD_sds = against_AD.std(axis=1)
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_cumulative_reward_game.csv', index_col=0)
        against_TFT_means = against_TFT.mean(axis=1)
        against_TFT_sds = against_TFT.std(axis=1)
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_cumulative_reward_game.csv', index_col=0)
        against_Random_means = against_Random.mean(axis=1)
        against_Random_sds = against_Random.std(axis=1)
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(against_AC.index[:], against_AC_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AC')#, color='red')
        plt.fill_between(against_AC.index[:], against_AC_means-against_AC_ci, against_AC_means+against_AC_ci, alpha=0.5) #facecolor='#ff9999', 
        plt.plot(against_AD.index[:], against_AD_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AD')#, color='#556b2f')
        plt.fill_between(against_AD.index[:], against_AD_means-against_AD_ci, against_AD_means+against_AD_ci, alpha=0.5) #facecolor='#ccff99', 
        plt.plot(against_TFT.index[:], against_TFT_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_TFT')#, color='#00cccc')
        plt.fill_between(against_TFT.index[:], against_TFT_means-against_TFT_ci, against_TFT_means+against_TFT_ci, alpha=0.5) #facecolor='#99ffff', 
        plt.plot(against_Random.index[:], against_Random_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_Random')#, color='orange')
        plt.fill_between(against_Random.index[:], against_Random_means-against_Random_ci, against_Random_means+against_Random_ci, alpha=0.5) #facecolor='#ffcc99', 
        
        plt.title(r'Cumulative game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player (baseline)')
        if game_title=='IPD':
            plt.gca().set_ylim([0, 40000])
        elif game_title=='VOLUNTEER':
            plt.gca().set_ylim([0, 50000])
        elif game_title=='STAGHUNT': 
            plt.gca().set_ylim([0, 50000])  
        plt.ylabel(f'Cumulative game reward for {player1_title}')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        if not os.path.isdir('results/outcome_plots/reward'):
            os.makedirs('results/outcome_plots/reward')
        plt.savefig(f'results/outcome_plots/reward/baseline_cumulative_game_reward_{player1_title}.png', bbox_inches='tight')


        ##################################
        #### non-cumulative - game reward for player1_tytle vs others  ####
        ##################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_reward_game.csv', index_col=0)
        against_AC_means = against_AC.mean(axis=1)
        against_AC_sds = against_AC.std(axis=1)
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_reward_game.csv', index_col=0)
        against_AD_means = against_AD.mean(axis=1)
        against_AD_sds = against_AD.std(axis=1)
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_reward_game.csv', index_col=0)
        against_TFT_means = against_TFT.mean(axis=1)
        against_TFT_sds = against_TFT.std(axis=1)
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_reward_game.csv', index_col=0)
        against_Random_means = against_Random.mean(axis=1)
        against_Random_sds = against_Random.std(axis=1)
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

        plt.figure(dpi=80) #figsize=(10, 6)
        plt.plot(against_AC.index[:], against_AC_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AC')#, color='red')
        plt.fill_between(against_AC.index[:], against_AC_means-against_AC_ci, against_AC_means+against_AC_ci, alpha=0.5) #facecolor='#ff9999', 
        plt.plot(against_AD.index[:], against_AD_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_AD')#, color='#556b2f')
        plt.fill_between(against_AD.index[:], against_AD_means-against_AD_ci, against_AD_means+against_AD_ci, alpha=0.5) #facecolor='#ccff99', 
        plt.plot(against_TFT.index[:], against_TFT_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_TFT')#, color='#00cccc')
        plt.fill_between(against_TFT.index[:], against_TFT_means-against_TFT_ci, against_TFT_means+against_TFT_ci, alpha=0.5) #facecolor='#99ffff', 
        plt.plot(against_Random.index[:], against_Random_means[:], lw=0.8, alpha=0.5, label=f'{player1_title}_Random')#, color='orange')
        plt.fill_between(against_Random.index[:], against_Random_means-against_Random_ci, against_Random_means+against_Random_ci, alpha=0.5) #facecolor='#ffcc99', 
        
        plt.title(r'Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ player1_title +' vs static player (baseline)')

        if game_title=='IPD':
            plt.gca().set_ylim([1, 4])
        elif game_title=='VOLUNTEER':
            plt.gca().set_ylim([1, 5])
        elif game_title=='STAGHUNT': 
            plt.gca().set_ylim([1, 5])
        plt.ylabel(f'Game reward for {player1_title}')
        plt.xlabel('Iteration')
        leg = plt.legend() # get the legend object
        for line in leg.get_lines(): # change the line width for the legend
            line.set_linewidth(4.0)
        plt.savefig(f'results/outcome_plots/reward/baseline_game_reward_{player1_title}.png', bbox_inches='tight')


    def plot_baseline_relative_reward(player1_title, n_runs):
        '''plot game reward - relatie cumulative (bar & over time) & per iteration
        - how well off did the players end up relative to each other on the game?'''
        ##################################
        #### bar chart game cumulative reward for player1_tytle vs others  ####
        ##################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_AC_means = against_AC.mean()
        against_AC_sds = against_AC.std()
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_AD_means = against_AD.mean()
        against_AD_sds = against_AD.std()
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_TFT_means = against_TFT.mean()
        against_TFT_sds = against_TFT.std()
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_cumulative_reward_game.csv', index_col=0).iloc[-1]
        against_Random_means = against_Random.mean()
        against_Random_sds = against_Random.std()
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)


        #import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(3, 3), dpi=80)
        plt.rcParams.update({'font.size':20})
        ax = fig.add_axes([0,0,1,1])
        labels = ['AC', 'AD', 'TFT', 'Rand.'] 
        means = [against_AC_means,against_AD_means,against_TFT_means,against_Random_means]
        cis = [against_AC_ci,against_AD_ci,against_TFT_ci,against_Random_ci]
        colors = ['lightgreen', 'crimson', 'gold', 'slategrey']  
        #add QLVM mixed agent 
        ax.bar(labels, means, yerr=cis, color=colors, width = 0.8) #capsize=7, 
        #plt.xticks(rotation=45)
        ax.set_ylabel(r'Cumulative $R_{extr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
        #(f'R^{extr}_{i=QLUT}'# for {player1_title}')
        long_title = title_mapping[player1_title].replace('Ethics','').replace('_', '-')
        ax.set_title(long_title +' vs other \n') #'Cumulative Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
        ax.set_xlabel('Opponent type')
        if game_title=='IPD': #NOTE game_title is set outside this function - in the overall environment - see code below 
            plt.gca().set_ylim([0, 40000])
        elif game_title=='VOLUNTEER':
            plt.gca().set_ylim([0, 50000])
        elif game_title=='STAGHUNT': 
            plt.gca().set_ylim([0, 50000])  
        if not os.path.isdir('results/outcome_plots/reward'):
            os.makedirs('results/outcome_plots/reward')
        plt.savefig(f'results/outcome_plots/reward/baseline_bar_cumulative_game_reward_{player1_title}.pdf', bbox_inches='tight')
        


        ##################################
        #### non-cumulative - game reward for player1_tytle vs others  ####
        ##################################
        figsize=(5, 4)

        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_reward_game.csv', index_col=0)
        against_AC_means = against_AC.mean(axis=1)
        against_AC_sds = against_AC.std(axis=1)
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_reward_game.csv', index_col=0)
        against_AD_means = against_AD.mean(axis=1)
        against_AD_sds = against_AD.std(axis=1)
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_reward_game.csv', index_col=0)
        against_TFT_means = against_TFT.mean(axis=1)
        against_TFT_sds = against_TFT.std(axis=1)
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_reward_game.csv', index_col=0)
        against_Random_means = against_Random.mean(axis=1)
        against_Random_sds = against_Random.std(axis=1)
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

    

        plt.figure(dpi=80, figsize=figsize)
        plt.rcParams.update({'font.size':20})
        plt.plot(against_AC.index[:], against_AC_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_AC', color='lightgreen')
        plt.fill_between(against_AC.index[:], against_AC_means-against_AC_ci, against_AC_means+against_AC_ci, facecolor='lightgreen', alpha=0.5)
        plt.plot(against_AD.index[:], against_AD_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_AD', color='crimson')
        plt.fill_between(against_AD.index[:], against_AD_means-against_AD_ci, against_AD_means+against_AD_ci, facecolor='crimson', alpha=0.5)
        plt.plot(against_TFT.index[:], against_TFT_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_TFT', color='yellow')
        plt.fill_between(against_TFT.index[:], against_TFT_means-against_TFT_ci, against_TFT_means+against_TFT_ci, facecolor='yellow', alpha=0.5)
        plt.plot(against_Random.index[:], against_Random_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_Random', color='darkgrey')
        plt.fill_between(against_Random.index[:], against_Random_means-against_Random_ci, against_Random_means+against_Random_ci, facecolor='darkgrey', alpha=0.5)


        #long_title = title_mapping[player1_title]
        plt.title(long_title +' vs other \n') #r'Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
        if game_title=='IPD':
            plt.gca().set_ylim([1, 4])
        elif game_title=='VOLUNTEER':
            plt.gca().set_ylim([1, 5])
        elif game_title=='STAGHUNT': 
            plt.gca().set_ylim([1, 5])
        plt.ylabel(r'$R_{extr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
        plt.xlabel('Iteration')
        #plt.xticks(rotation=45)
        #leg = plt.legend() # get the legend object
        #for line in leg.get_lines(): # change the line width for the legend
        #    line.set_linewidth(4.0)
        plt.savefig(f'results/outcome_plots/reward/baseline_game_reward_{player1_title}.pdf', bbox_inches='tight')

    def plot_baseline_relative_moral_reward(player1_title, n_runs):
        '''plot moral reward - relatie cumulative (over time & bar plot) & per iteration
        - how well off did the players end up relative to each other in terms of moral reward?'''
        
        ##################################
        #### bar chart moral cumulative reward for player1_tytle vs others  ####
        ##################################
        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_AC_means = against_AC.mean()
        against_AC_sds = against_AC.std()
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_AD_means = against_AD.mean()
        against_AD_sds = against_AD.std()
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_TFT_means = against_TFT.mean()
        against_TFT_sds = against_TFT.std()
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_cumulative_reward_intrinsic.csv', index_col=0).iloc[-1]
        against_Random_means = against_Random.mean()
        against_Random_sds = against_Random.std()
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

        #import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(3, 3), dpi=80)
        plt.rcParams.update({'font.size':20})
        ax = fig.add_axes([0,0,1,1])
        labels = ['AC', 'AD', 'TFT', 'Rand.'] 
        means = [against_AC_means,against_AD_means,against_TFT_means,against_Random_means]
        cis = [against_AC_ci,against_AD_ci,against_TFT_ci,against_Random_ci]
        colors = ['lightgreen', 'crimson', 'gold', 'slategrey']  
        #add QLVM mixed agent 
        ax.bar(labels, means, yerr=cis, color=colors, width = 0.8) #capsize=7, 
        #plt.xticks(rotation=45)
        ax.set_ylabel(r'Cumulative $R_{intr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
        #(f'R^{extr}_{i=QLUT}'# for {player1_title}')
        long_title = title_mapping[player1_title].replace('Ethics','').replace('_', '-')
        ax.set_title(long_title +' vs other \n') #'Cumulative Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
        ax.set_xlabel('Opponent type')
        if game_title=='IPD':
            if player1_title == 'QLUT':
                plt.gca().set_ylim([0, 60000])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-50000, 0]) #[-1300, 0]
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 10000])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 50000])
            elif player1_title == 'QLVM':
                plt.gca().set_ylim([0, 10000])
        elif game_title=='VOLUNTEER':
            if player1_title == 'QLUT':
                plt.gca().set_ylim([0, 80000])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-50000, 0]) #[-1300, 0]
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 10000])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 50000])
            elif player1_title == 'QLVM':
                plt.gca().set_ylim([0, 10000])
        elif game_title=='STAGHUNT': 
            if player1_title == 'QLUT':
                plt.gca().set_ylim([0, 100000])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-50000, 0]) #[-1300, 0]
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 10000])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 50000])
            elif player1_title == 'QLVM':
                plt.gca().set_ylim([0, 10000])
        plt.savefig(f'results/outcome_plots/reward/baseline_bar_cumulative_intrinsic_reward_{player1_title}.pdf', bbox_inches='tight')
        
        
        ##################################
        #### non-cumulative - game reward for player1_tytle vs others  ####
        ##################################
        figsize=(5, 4)

        against_AC = pd.read_csv(f'results/{player1_title}_AC/player1/df_reward_intrinsic.csv', index_col=0)
        against_AC_means = against_AC.mean(axis=1)
        against_AC_sds = against_AC.std(axis=1)
        against_AC_ci = 1.96 * against_AC_sds/np.sqrt(n_runs)

        against_AD = pd.read_csv(f'results/{player1_title}_AD/player1/df_reward_intrinsic.csv', index_col=0)
        against_AD_means = against_AD.mean(axis=1)
        against_AD_sds = against_AD.std(axis=1)
        against_AD_ci = 1.96 * against_AD_sds/np.sqrt(n_runs)

        against_TFT = pd.read_csv(f'results/{player1_title}_TFT/player1/df_reward_intrinsic.csv', index_col=0)
        against_TFT_means = against_TFT.mean(axis=1)
        against_TFT_sds = against_TFT.std(axis=1)
        against_TFT_ci = 1.96 * against_TFT_sds/np.sqrt(n_runs)

        against_Random = pd.read_csv(f'results/{player1_title}_Random/player1/df_reward_intrinsic.csv', index_col=0)
        against_Random_means = against_Random.mean(axis=1)
        against_Random_sds = against_Random.std(axis=1)
        against_Random_ci = 1.96 * against_Random_sds/np.sqrt(n_runs)

    

        plt.figure(dpi=80, figsize=figsize)
        plt.rcParams.update({'font.size':20})
        plt.plot(against_AC.index[:], against_AC_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_AC', color='lightgreen')
        plt.fill_between(against_AC.index[:], against_AC_means-against_AC_ci, against_AC_means+against_AC_ci, facecolor='lightgreen', alpha=0.5)
        plt.plot(against_AD.index[:], against_AD_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_AD', color='crimson')
        plt.fill_between(against_AD.index[:], against_AD_means-against_AD_ci, against_AD_means+against_AD_ci, facecolor='crimson', alpha=0.5)
        plt.plot(against_TFT.index[:], against_TFT_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_TFT', color='yellow')
        plt.fill_between(against_TFT.index[:], against_TFT_means-against_TFT_ci, against_TFT_means+against_TFT_ci, facecolor='yellow', alpha=0.5)
        plt.plot(against_Random.index[:], against_Random_means[:], lw=0.1, alpha=0.5, label=f'{player1_title}_Random', color='darkgrey')
        plt.fill_between(against_Random.index[:], against_Random_means-against_Random_ci, against_Random_means+against_Random_ci, facecolor='darkgrey', alpha=0.5)


        #long_title = title_mapping[player1_title]
        plt.title(long_title +' vs other \n') #r'Game Reward (Mean '+r'$\pm$ 95% CI over ' +str(n_runs)+' runs), '+'\n'+ 
        if game_title=='IPD':
            if player1_title == 'QLUT':
                plt.gca().set_ylim([4, 6])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-5, 0]) 
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 1])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 5])
        elif game_title=='VOLUNTEER':
            if player1_title == 'QLUT':
                plt.gca().set_ylim([2, 8])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-5, 0]) 
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 1])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 5])
        elif game_title=='STAGHUNT': 
            if player1_title == 'QLUT':
                plt.gca().set_ylim([4, 10])
            elif player1_title == 'QLDE':
                plt.gca().set_ylim([-5, 0]) 
            elif player1_title == 'QLVE_e':
                plt.gca().set_ylim([0, 1])
            elif player1_title == 'QLVE_k':
                plt.gca().set_ylim([0, 5])
        plt.ylabel(r'$R_{intr}$ for '+str(player1_title).replace('QL','').replace('_e', r'$_e$').replace('_k', r'$_k$').replace('M', 'E'+r'$_m$'))
        plt.xlabel('Iteration')
        #plt.xticks(rotation=45)
        #leg = plt.legend() # get the legend object
        #for line in leg.get_lines(): # change the line width for the legend
        #    line.set_linewidth(4.0)
        plt.savefig(f'results/outcome_plots/reward/baseline_intrinsic_reward_{player1_title}.pdf', bbox_inches='tight')


########################
#### plots per episode ####
########################


def episode_plot_cooperative_selections_whereavailable(destination_folder, titles, n_runs):
    '''create cooperative_selection bool variable for each run, then plot % cooperative selections'''
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')
    
    runnames_intermediate = ['run'+str(i+1) for i in range(n_runs)]
    data_coop_selections = pd.DataFrame(columns=runnames_intermediate)
    data_selections = pd.DataFrame(columns=runnames_intermediate)
    data_availables = pd.DataFrame(columns=runnames_intermediate)

    path = str(destination_folder+'/history/'+'*.csv')
        
    for run in glob.glob(path): #for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv').split('/')[-1]
        try: 
            run_df = pd.read_csv(str(run), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
            #drop NAs from the df - i.e. drop rows where a selection was not actually made 
            run_df_clean = run_df[run_df['selected_prev_move'].notna()]
            
            #create boolean variable for cooperative selections amde 
            run_df_clean['cooperative_sel_made'] = run_df_clean.apply(lambda row: row['selected_prev_move']==0, axis=1)

            #create boolean variable for cooperative partners available using env_state 
            run_df_clean['cooperative_partners_available_new'] = run_df_clean.apply(lambda row: row['cooperative_sel_available']==True, axis=1)

            #group by episode 
            run_grouped = run_df_clean[['episode', 'cooperative_partners_available_new', 'cooperative_sel_made']].groupby('episode').agg({
                'cooperative_partners_available_new': 'sum', 'cooperative_sel_made': 'sum'})

            run_grouped['%_coop_sel'] = run_grouped['cooperative_sel_made'] * 100 / run_grouped['cooperative_partners_available_new']
            #fill in 0-division as 0 --> does this make sense?? 
            #run_grouped['%_coop_sel'] = run_grouped.apply(lambda row: 0 if row['cooperative_sel_made']==0 and row['cooperative_sel_available']==0, axis=1)

            data_coop_selections[run_name] = run_grouped['%_coop_sel']
            data_selections[run_name] = run_grouped['cooperative_sel_made']
            data_availables[run_name] = run_grouped['cooperative_partners_available_new']

        except:
            print('failed at run :', run_name)
            break 

    mean_coop_sel = data_coop_selections.mean(axis=1)
    sd_coop_sel = data_coop_selections.std(axis=1)
    ci = 1.96 * sd_coop_sel/np.sqrt(n_runs)

    mean_sel = data_selections.mean(axis=1)
    mean_avail = data_availables.mean(axis=1)

    if len(titles) < 3:
        if 'AC' in titles: 
            linewidth=0.9
        elif 'AD' in titles:
            linewidth=0.9
        else: 
            linewidth = 0.9
    else: 
        linewidth = 0.07

    #plot results 
    plt.figure(dpi=80, figsize=(15, 6))
    #plt.plot(result.index[:], result[:], label=r'% cooperative selections', color='blue', linewidth=linewidth)
    plt.plot(mean_coop_sel, label=r'mean % coop. selections', color='blue', linewidth=linewidth)
    plt.fill_between(mean_coop_sel.index[:], mean_coop_sel-ci, mean_coop_sel+ci, facecolor='lightblue', alpha=0.9)
    plt.title(f'Cooperative selections in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-5, 105])
    plt.ylabel("% cooperative partners selected \n (those that C in last move) \n over all cooperative partners available  \n (mean over "+str(n_runs)+f" runs +- 95% CI")
    plt.xlabel('Episode')

    #save the plot
    if not os.path.isdir(f'{destination_folder}/plots'):
            os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_cooperative_selections_{population_list}_whereavailable.png', bbox_inches='tight')

    population_learners = [x for x in titles if x not in ['AC', 'AD', 'TFT', 'Random', 'CD', 'GEa','GEb', 'GEc', 'GEd', 'GEe']]
    
    #plot cooperative selections made - absolute number out of n_runs
    plt.figure(dpi=80, figsize=(15, 6))
    #plt.plot(data_selections['sum_across_runs'].index[:], data_selections['sum_across_runs'][:], label=r'num cooperative selections made', color='blueviolet', linewidth=0.07)
    plt.plot(mean_sel, label=r'num cooperative selections made', color='blueviolet', linewidth=linewidth)
    plt.title(f'Cooperative selections made in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-0.1, len(population_learners)+0.1]) #plt.gca().set_ylim([-0.5, n_runs+0.5])
    plt.ylabel("num cooperative selections made  \n (those that C in last move)  \n (mean over "+str(n_runs)+r" runs)")
    plt.xlabel('Episode')

    if ('AC' in titles or 'AD' in titles): 
        linewidth=0.9
    else: 
        linewidth = 0.07

    #plot cooperative selections available - absolute number out of n_runs
    plt.figure(dpi=80, figsize=(15, 6))
    #plt.plot(data_availables['sum_across_runs'].index[:], data_availables['sum_across_runs'][:], label=r'num cooperative opponents available', color='indigo', linewidth=0.07)
    plt.plot(mean_avail, label=r'num cooperative opponents available', color='indigo', linewidth=linewidth)
    plt.title(f'Cooperative partners available in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-0.1, len(population_learners)+0.1]) #plt.gca().set_ylim([-0.5, n_runs+0.5])
    plt.ylabel("num cooperative partners available \n (those that C in last move)  \n (mean over "+str(n_runs)+r" runs)")
    plt.xlabel('Episode')

def episode_plot_cooperative_selections_perrun(destination_folder, titles, n_runs):
    '''create cooperative_selection bool variable for each run, then plot % cooperative selections'''
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')
    
    runnames_intermediate = ['run'+str(i+1) for i in range(n_runs)]
    data_coop_selections = pd.DataFrame(columns=runnames_intermediate)
    data_selections = pd.DataFrame(columns=runnames_intermediate)
    data_availables = pd.DataFrame(columns=runnames_intermediate)

    path = str(destination_folder+'/history/'+'*.csv')
        
    for run in glob.glob(path): #for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv').split('/')[-1]
        try: 
            run_df = pd.read_csv(str(run), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
            #drop NAs from the df - i.e. drop rows where a selection was not actually made 
            run_df_clean = run_df[run_df['selected_prev_move'].notna()]
            
            #create boolean variable for cooperative selections amde 
            run_df_clean['cooperative_sel_made'] = run_df_clean.apply(lambda row: row['selected_prev_move']==0, axis=1)

            #create boolean variable for cooperative partners available using env_state 
            run_df_clean['cooperative_partners_available_new'] = run_df_clean.apply(lambda row: row['cooperative_sel_available']==True, axis=1)

            #group by episode 
            run_grouped = run_df_clean[['episode', 'cooperative_partners_available_new', 'cooperative_sel_made']].groupby('episode').agg({
                'cooperative_partners_available_new': 'sum', 'cooperative_sel_made': 'sum'})

            run_grouped['%_coop_sel'] = run_grouped['cooperative_sel_made'] * 100 / run_grouped['cooperative_partners_available_new']
            #fill in 0-division as 0 --> does this make sense?? 
            #run_grouped['%_coop_sel'] = run_grouped.apply(lambda row: 0 if row['cooperative_sel_made']==0 and row['cooperative_sel_available']==0, axis=1)

            data_coop_selections[run_name] = run_grouped['%_coop_sel']
            data_selections[run_name] = run_grouped['cooperative_sel_made']
            data_availables[run_name] = run_grouped['cooperative_partners_available_new']

        except:
            print('failed at run :', run_name)
            break 

    
    for run in data_coop_selections.columns:
        plt.figure(dpi=80, figsize=(15, 6))
        plt.plot(data_coop_selections[run], linewidth=0.08)# alpha=0.5
        plt.title(f'Cooperative selections in population \n {population_list}, {run}') 
        #plt.gca().set_ylim([-5, 105])
        plt.ylabel("% cooperative partners selected \n (those that C in last move) \n over all cooperative partners available  \n (every run separate")
        plt.xlabel('Episode')

    




def episode_plot_cooperation_population_v2(destination_folder, titles, n_runs, num_iter, with_CI=True, reduced=False):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating? 
    if reduced==True, plot every 5th value'''
    
    if len(titles)<=6:
        titles_short = [title.replace('QL','') for title in titles] 
        temp = ''
        for t in titles_short: 
            temp += (t+',')
        population_list = temp.strip(',')
    else:  
        population_list = destination_folder.replace('results/', '')

    #combine each player's actions into one df: 
    result_population = pd.DataFrame(index=range(num_iter)) 

    for run_idx in range(n_runs):
        run_idx += 1
        run_df = pd.read_csv(destination_folder+f'/history/run{run_idx}.csv', index_col=0)[['episode', 'action_player1', 'action_player2']]

        #create boolean variabnle for cooperative selections amde 
        run_df['action_player1_C'] = run_df.apply(lambda row: row['action_player1']==0, axis=1)
        run_df['action_player2_C'] = run_df.apply(lambda row: row['action_player2']==0, axis=1)

        #group by episode 
        run_grouped = run_df.groupby('episode').agg({'action_player1_C': 'sum', 'action_player2_C': 'sum', 'action_player1':'count', 'action_player2':'count'})
        run_grouped['action_C'] = run_grouped['action_player1_C'] + run_grouped['action_player2_C']
        run_grouped['actions_total'] = run_grouped['action_player1'] + run_grouped['action_player2']
        #TO DO check the denominator is 40 
        run_grouped['%_C'] = run_grouped['action_C'] * 100 / run_grouped['actions_total']

        result_population[f'run{run_idx}'] = run_grouped['%_C']


    if with_CI:
        means = result_population.mean(axis=1)
        sds = result_population.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        result_population['%_cooperate_mean'] = means
        result_population['%_cooperate_ci'] = ci
    else: 
        result_population['%_cooperate_mean'] = result_population.mean(axis=1)

    if reduced: 
        #keep every 5th row 
        result_population = result_population[result_population.index % 5 == 0]
        reduced_option = ' (every 5)'
    else: 
        reduced_option = None

    #plot results 
    plt.figure(dpi=80, figsize=(15, 6)) 
    if with_CI:
                plt.plot(result_population.index[:], result_population['%_cooperate_mean'], label=f'{population_list}', color='darkgreen', linewidth=0.05)
                plt.fill_between(result_population.index[:], result_population['%_cooperate_mean']-result_population['%_cooperate_ci'], result_population['%_cooperate_mean']+result_population['%_cooperate_ci'], facecolor='lightgreen', linewidth=0.04, alpha=1)
    else: 
        plt.plot(result_population.index[:], result_population['%_cooperate_mean'], label=f'{population_list}', color='darkgreen', linewidth=0.05)

    if reduced_option: 
        plt.title(f'Cooperation in population \n {population_list} '+ reduced_option) 
    else: 
        plt.title(f'Cooperation in population \n {population_list} ') 
    plt.gca().set_ylim([0, 100])
    plt.ylabel(r'% cooperating per episode'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')


    if not os.path.isdir(f'{destination_folder}/plots'):
        os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_cooperation_{population_list}.png', bbox_inches='tight')

def episode_plot_cooperation_perrun(destination_folder, titles, n_runs, num_iter):
    
    if len(titles)<=6:
        titles_short = [title.replace('QL','') for title in titles] 
        temp = ''
        for t in titles_short: 
            temp += (t+',')
        population_list = temp.strip(',')
    else:  
        population_list = destination_folder.replace('results/', '')

    #combine each player's actions into one df: 
    result_population = pd.DataFrame(index=range(num_iter)) 

    for run_idx in range(n_runs):
        run_idx += 1
        run_df = pd.read_csv(destination_folder+f'/history/run{run_idx}.csv', index_col=0)[['episode', 'action_player1', 'action_player2']]

        #create boolean variabnle for cooperative selections amde 
        run_df['action_player1_C'] = run_df.apply(lambda row: row['action_player1']==0, axis=1)
        run_df['action_player2_C'] = run_df.apply(lambda row: row['action_player2']==0, axis=1)

        #group by episode 
        run_grouped = run_df.groupby('episode').agg({'action_player1_C': 'sum', 'action_player2_C': 'sum', 'action_player1':'count', 'action_player2':'count'})
        run_grouped['action_C'] = run_grouped['action_player1_C'] + run_grouped['action_player2_C']
        run_grouped['actions_total'] = run_grouped['action_player1'] + run_grouped['action_player2']
        #TO DO check the denominator is 40 
        run_grouped['%_C'] = run_grouped['action_C'] * 100 / run_grouped['actions_total']

        result_population[f'run{run_idx}'] = run_grouped['%_C']

    #plot results 
    for run in result_population.columns:

        plt.figure(dpi=80, figsize=(15, 6)) 
        plt.plot(result_population.index[:], result_population[run], label=f'{population_list}', color='darkgreen', linewidth=0.05)
        plt.title(f'Cooperation in population \n {population_list} ') 
        plt.gca().set_ylim([0, 100])
        plt.ylabel(r'% cooperating per episode'+ '  \n (every run separate)')
        plt.xlabel('Episode')  



def episode_plot_action_pairs_population_new(destination_folder, titles, n_runs, option=None, combine_CDDC=False):
    '''if reduced, plot only every 10th value. 
       if combine_CDDC, ignore order of action pairs, and plot CD and DC (exploitation in one direction) together. This ignores whether the selector or the selected is the defector.'''
    if not os.path.isdir(str(destination_folder)+'/plots'):
        os.makedirs(str(destination_folder)+'/plots')

    if not os.path.isdir(str(destination_folder)+'/plots/outcome_plots'):
        os.makedirs(str(destination_folder)+'/plots/outcome_plots')

    if not os.path.isdir('results/outcome_plots/actions'):
            os.makedirs('results/outcome_plots/actions')


    if 'action_pairs.csv' not in os.listdir(str(destination_folder+'/')): 
    #if True: 
        colnames = ['run'+str(i+1) for i in range(n_runs)]
        #append 'episode' to the end of the list 
        #colnames.append('episode')
        actions_player1 = pd.DataFrame(columns=colnames, index=range(num_iter*population_size_approx))
        actions_player2 = pd.DataFrame(columns=colnames,  index=range(num_iter*population_size_approx))

        print('gathering actions from each run ...')
        for run in os.listdir(str(destination_folder+'/history')):
            if '.csv' in run:
                run_name = str(run).strip('.csv')
                run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
                actions_player1[run_name] = run_df['action_player1']
                actions_player1['episode'] = run_df['episode']
                actions_player2[run_name] = run_df['action_player2']
                #actions_player2['episode'] = run_df['episode']

        results = pd.DataFrame(columns=colnames)

        print('combining actions from each run into pairs ...')
        for colname in colnames: #loop over every run
            str_value = actions_player1[colname].astype(str) + ',' + actions_player2[colname].astype(str)
            str_value = str_value.str.replace('1', 'D')
            str_value = str_value.str.replace('0', 'C')
            str_value = str_value.str.replace(' ', '')
            if combine_CDDC:
                str_value = str_value.replace('C,D', 'C,D/D,C').replace('D,C', 'C,D/D,C') #when the order of actions does not matter, combine CD/DC into one color
            results[colname] = str_value

        #results['episode'] = actions_player1['episode']

        print('storing action_pairs to disc...')
        results.to_csv(str(destination_folder+'/action_pairs.csv'))
    else: 
        print('importing existing action_pairs file...')
        results = pd.read_csv(str(destination_folder+'/action_pairs.csv'), index_col=0)


    print('calculating value_counts for each action pair...')
    results_counts = results.apply(lambda x: x.value_counts(), axis=1)
    #results_counts = results.transpose().apply(pd.value_counts).transpose()#[1:] #plot after first episode, as in the first apisode they are reacting to default state=0
    results_counts['episode'] = pd.read_csv(str(destination_folder+'/history/run1.csv'), index_col=0)['episode'] #actions_player1['episode']
    results_counts_grouped = results_counts.groupby('episode').sum() #sum all the counts of each pair across the entire episode
    #y-axis = number of runs * population_size (i.e., action pairs in a single episode)

    #results_counts_grouped.dropna(axis=1, how='all', inplace=True)

    titles = [title.replace('QL','') for title in titles] 
    titles = [title.replace('Ethics','').replace('_','-').replace('VE-e','VE_e').replace('VE-k','VE_k') for title in titles] 
    temp = ''
    for t in titles: 
        temp += (t+',')
    temp = temp.strip(',')
    if len(titles)>6: 
        #temp = destination_folder.replace('results/', '').replace('__iter30000_runs20', '')
        most_common_numbers = Counter(titles).most_common(2)
        if most_common_numbers[0][1] == most_common_numbers[1][1]:
            print('more than one majority value')
            population_forpaper = f'uniform'
            popsize_forpaper = f', size {len(titles)}'
        else: 

            #population = destination_folder.replace('___iter30000_runs20', '').replace('results/', '')
            #long_titles = get_titles_for_population(destination_folder)
            majority_title_short = most_common_numbers[0][0] #max(set(titles), key = titles.count)
            majority_title = title_mapping['QL'+majority_title_short]
            population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'
            popsize_forpaper = ''

    #plt.figure(figsize=(20, 15), dpi=100)
    if num_iter > 20000:
        plt.figure(dpi=80, figsize=(10, 4))
    else:
        plt.figure(dpi=80, figsize=(5, 4))
    
    if combine_CDDC:
        color = {'C,C':'#28641E', 'C,D/D,C':'#FFFFFF', 'D,D':'#8E0B52'}
    else: 
        color = {'C,C':'#28641E', 'C,D':'#B0DC82', 'D,C':'#EEAED4', 'D,D':'#8E0B52'}

    print('plotting ...')
    sns.set(font_scale=3.5)
    #plot as an area plot 
    total_actionpairs = results_counts_grouped.sum(axis=1)[0]
    #plt.rcParams.update({'font.size':20})
    results_counts_grouped.plot.area(stacked=True, ylabel = '# action pairs \n (over 20 runs)', #rot=45,
        xlabel='Episode', #each episode includes N iterations, where N = population_size (eahc player makes a selection once)
        xticks=[0,15000,30000], yticks=[0,total_actionpairs/2,total_actionpairs],
        color=color, linewidth=0.01, alpha=0.9,
        #color={'C, C':'#28641E', 'C, D':'#B0DC82', 'D, C':'#EEAED4', 'D, D':'#8E0B52'}, linewidth=0.05, alpha=0.9,
        #color={'C, C':'royalblue', 'C, D':'lightblue', 'D, C':'yellow', 'D, D':'orange'},
        title = str('Action pairs: \n ' + population_forpaper + ' population'+  popsize_forpaper)
        ).legend(fontsize=18, loc='center') #Pairs of simultaneous actions over time: \n '+

    #plt.savefig(f'{destination_folder}/plots/action_pairs.png', bbox_inches='tight')
    if not option: #if plotting main results
        population = destination_folder.split('/')[1]
    else: #if plotting extra parameter search for beta in QLVM 
        population = destination_folder
        #pair += str(option)
    
    plt.savefig(f'results/outcome_plots/actions/episode_actions_pairs_popul{population_forpaper}.png', bbox_inches='tight')



def episode_plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, num_iter, population_size):
    '''explore the actions taken by players aggregated per episode - what percentage are cooperating? '''
    
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')

    if False: 
        if population_size == 2: 
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.06
        elif population_size == 3: 
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.05
        elif population_size == 4: 
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.03
        elif population_size == 5: 
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.01
        elif population_size == 6: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=1, ncols=6, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.01
        elif population_size == 7: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(nrows=1, ncols=7, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.01
        elif population_size == 10: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005
        elif population_size == 14: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14) = plt.subplots(nrows=1, ncols=14, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005
        elif population_size == 20: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005 #0.0005
        elif population_size == 11: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11) = plt.subplots(nrows=1, ncols=11, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005
        elif population_size == 21: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21) = plt.subplots(nrows=1, ncols=21, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005
        
    fig, axs = plt.subplots(nrows=1, ncols=population_size, sharey=True, sharex=True, figsize=(3*population_size,2.5))
    linewidth = 0.005
    #results_idx = pd.DataFrame(index=range(num_iter))

    episodes_column = np.repeat(range(num_iter), population_size)

    #plot for each player type: 
    for idx in range(len(titles)):
        title = title_mapping[titles[idx]] #get long title 

        if title in ['AlwaysCooperate', 'AlwaysDefect', 'GilbertElliot_typeb']:
            linewidth_new = 1.5
        else: 
            linewidth_new = linewidth

        actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
        results_idx = pd.DataFrame(index=range(num_iter)) #actions_idx.index)

        #manually add episode to the actions_df 
#TO DO check that this will work correctly in all cases and with NAs in Selfish player's actions !!! 

        #mask D actions, only keeping Cc so we can the count 
        actions_C = actions_idx.mask(actions_idx.ne('C'))
        actions_C['episode'] = episodes_column
        
        actions_C_episode = actions_C.groupby('episode').count()

        actions_idx['episode'] = episodes_column
        actions_all_episode = actions_idx.groupby('episode').count()
        

        #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
        results_idx['%_cooperate'] = actions_C_episode.sum(axis='columns') / actions_all_episode.sum(axis='columns') * 100


        #ax = locals()["ax"+str(idx+1)]
        ax = axs[idx]
        ax.set_title(f'\n {title} \n player{idx}') #f'\n {title} player{idx}'
        #plt.title(f'Cooperation by {title} player{idx} \n from population {population_list}') #(\n percent cooperated over '+str(n_runs)+r' runs)'

        if movingaverage: 
            linewidth_new = 0.5
            movingaverage_option = f'\n (moving average, window {movingN})'

            movMean = results_idx.rolling(window=movingN,center=True,min_periods=1).mean()
            #movStd = results_idx.rolling(window=movingNN,center=True,min_periods=1).std()
            ## get mean +/- 1.96*std/sqrt(N)
            #confIntPos = movMean + 1.96 * movStd / np.sqrt(N)
            #confIntNeg = movMean - 1.96 * movStd / np.sqrt(N)

            ax.plot(results_idx.index[:], movMean, color='darkgreen', linewidth=linewidth_new) #label=f'{title}_{idx}', 

        else: #if plotting every value 
            movingaverage_option = ''
            #plot results 
            ax.plot(results_idx.index[:], results_idx['%_cooperate'], color='darkgreen', linewidth=linewidth_new) #label=f'{title}_{idx}', 
        
        ax.set_ylim([-10, 110])
        ax.set(xlabel='Episode', ylabel='') #ylabel='Percentage cooperating \n (over '+str(n_runs)+r' runs)'

        #plt.savefig(f'{destination_folder}/plots/cooperation_{title}_{idx}.png', bbox_inches='tight')

    axs[0].set(ylabel=r'% cooperating'+ f'\n(over {n_runs} runs) {movingaverage_option}')
    #plt.ylabel(r'% cooperating'+ f'\n(over {n_runs} runs)')
    plt.suptitle(f'% cooperating for each player from population {population_list} \n', fontsize=28, y=1.4)

    if not os.path.isdir(f'{destination_folder}/plots'):
        os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_cooperation_eachplayer_{movingaverage_option}.png', bbox_inches='tight')



def episode_plot_reward_eachplayerinpopulation(destination_folder, titles, n_runs, num_iter, population_size, which='game', first_100=False):
    '''explore the actions taken by players aggregated per episode - what percentage are cooperating? '''
    
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')

    if which == 'intrinsic':
        sharey = False
    elif which == 'game':
        sharey=True
    fig, axs = plt.subplots(nrows=1, ncols=population_size, sharey=sharey, sharex=True, figsize=(3*population_size,2.5))
    #linewidth = 0.0007
    linewidth = 0.001

    #results_idx = pd.DataFrame(index=range(num_iter))

    episodes_column = np.repeat(range(num_iter), population_size)

    #plot for each player type: 
    for idx in range(len(titles)):
        title = title_mapping[titles[idx]] #get long title 

        if title in ['AlwaysCooperate', 'AlwaysDefect', 'GilbertElliot_typeb']:
            linewidth_new = 1.5
        elif first_100:
            linewidth_new = 0.5
        else: 
            linewidth_new = linewidth


        if first_100:
            results_idx = pd.DataFrame(index=range(100)) #actions_idx.index)
        else:
            results_idx = pd.DataFrame(index=range(num_iter*population_size)) #actions_idx.index)


        for run in range(1, n_runs+1): 
            if first_100:
                run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0).sort_index()[0:100]
            else:
                run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0).sort_index()
            small_run_df_p1 = run_df[run_df['idx_p1']==idx]
            rewards_idx_p1 = small_run_df_p1[[f'reward_{which}_player1']]
            small_run_df_p2 = run_df[run_df['idx_p2']==idx]
            rewards_idx_p2 = small_run_df_p2[[f'reward_{which}_player2']]

            rewards_idx = rewards_idx_p1[f'reward_{which}_player1'].append(rewards_idx_p2[f'reward_{which}_player2'])
            rewards_idx.sort_index(inplace=True)
            rewards_idx_cumsum = np.cumsum(rewards_idx)
            results_idx[f'run{run}'] = rewards_idx_cumsum

        #calculate average across runs for every iteration 
        means_reward = results_idx.mean(axis='columns')
        sds_reward =  results_idx.std(axis='columns')
        CIs_reward = 1.96 * sds_reward/np.sqrt(n_runs)

        #results_idx['episode'] = episodes_column
        #results_idx_episode = results_idx.groupby('episode').count()

        #### TO EDIT FROM HERE
        if False: 
            #mask D actions, only keeping Cc so we can the count 
            actions_C = actions_idx.mask(actions_idx.ne('C'))
            actions_C['episode'] = episodes_column
            
            actions_C_episode = actions_C.groupby('episode').count()

            actions_idx['episode'] = episodes_column
            actions_all_episode = actions_idx.groupby('episode').count()
            

            #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
            results_idx['%_cooperate'] = actions_C_episode.sum(axis='columns') / actions_all_episode.sum(axis='columns') * 100

        #ax = locals()["ax"+str(idx+1)]
        ax = axs[idx]
        ax.set_title(f'\n {title} \n player{idx}') #f'\n {title} player{idx}'
        #plt.title(f'Rewards for {title} player{idx} \n from population {population_list}') #(\n percent cooperated over '+str(n_runs)+r' runs)'

        #plot results 
        ax.plot(results_idx.index[:], means_reward, color='blue', linewidth=linewidth_new, alpha=0.5) #label=f'{title}_{idx}', 
        ax.fill_between(results_idx.index[:], means_reward-CIs_reward, means_reward+CIs_reward, facecolor='lightblue', alpha=0.5)

        #ax.set_ylim([-5, num_iter*population_size*4])
        ax.set(xlabel='Iteration', ylabel='') #ylabel='Percentage cooperating \n (over '+str(n_runs)+r' runs)'

        #plt.savefig(f'{destination_folder}/plots/cooperation_{title}_{idx}.png', bbox_inches='tight')

    axs[0].set(ylabel=f'Cumulative {which[0].upper()+which[1:]} Reward \n(mean over {n_runs} runs)')
    #plt.ylabel(r'% cooperating'+ f'\n(over {n_runs} runs)')
    if first_100:
        string = 'FIRST 100 -'
    else:
        string=''
    plt.suptitle(f'{string}Cumulative {which[0].upper()+which[1:]} Reward for each player from population {population_list} \n', fontsize=28, y=1.4)

    plt.savefig(f'{destination_folder}/plots/episode_cumul{which}reward_eachplayer.png', bbox_inches='tight')


def episode_plot_collective_outcomes_population(destination_folder, titles, n_runs, num_iter, population_size):
    '''explore the actions taken by players aggregated per episode - what percentage are cooperating? '''
    
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')

 
    results_collective_reward = pd.DataFrame(index=range(num_iter))
    results_collective_reward_cumulative = pd.DataFrame(index=range(num_iter))
    results_gini_reward = pd.DataFrame(index=range(num_iter))
    results_min_reward = pd.DataFrame(index=range(num_iter))

    
    for run_idx in range(n_runs):
        run_idx += 1
        run_df = pd.read_csv(destination_folder+f'/history/run{run_idx}.csv', index_col=0)[['episode', 'reward_collective', 'reward_gini', 'reward_min']]

        #group by episode 
        run_grouped = run_df.groupby('episode').agg({'reward_collective': 'sum', 'reward_gini':'mean', 'reward_min':'mean'})

        results_collective_reward[f'run{run_idx}'] = run_grouped['reward_collective']
        results_collective_reward_cumulative[f'run{run_idx}'] = np.cumsum(run_grouped['reward_collective'])
        results_gini_reward[f'run{run_idx}'] = run_grouped['reward_gini']
        results_min_reward[f'run{run_idx}'] = run_grouped['reward_min']

    #do COLLECTIVE_REWARD first - non-cumulative 
    results_collective_reward['collective_R_mean'] = results_collective_reward.mean(axis=1)
    sds = results_collective_reward.std(axis=1)
    results_collective_reward['collective_R_ci'] = 1.96 * sds/np.sqrt(n_runs)

    #plot results - non-cumulative 
    plt.figure(dpi=80, figsize=(15, 6)) 
    plt.plot(results_collective_reward.index[:], results_collective_reward['collective_R_mean'], label=f'{population_list}', color='purple', linewidth=0.05)
    plt.fill_between(results_collective_reward.index[:], results_collective_reward['collective_R_mean']-results_collective_reward['collective_R_ci'], results_collective_reward['collective_R_mean']+results_collective_reward['collective_R_ci'], facecolor='lightpink', linewidth=0.04, alpha=1)
   
    plt.title(f'Collective reward (mean) in population \n {population_list} ') 
    plt.gca().set_ylim([2*population_size, 6*population_size])
    plt.ylabel(r'Collective reward (mean per episode)'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')
    
    
    if not os.path.isdir(f'{destination_folder}/plots'):
        os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_collective_R_{population_list}.png', bbox_inches='tight')


    #do COLLECTIVE_REWARD - cumulative 
    results_collective_reward_cumulative['collective_R_mean'] = results_collective_reward_cumulative.mean(axis=1)
    sds = results_collective_reward_cumulative.std(axis=1)
    results_collective_reward_cumulative['collective_R_ci'] = 1.96 * sds/np.sqrt(n_runs)

    #plot results - cumulative 
    plt.figure(dpi=80, figsize=(15, 6)) 
    plt.plot(results_collective_reward_cumulative.index[:], results_collective_reward_cumulative['collective_R_mean'], label=f'{population_list}', color='purple', linewidth=0.05)
    plt.fill_between(results_collective_reward_cumulative.index[:], results_collective_reward_cumulative['collective_R_mean']-results_collective_reward_cumulative['collective_R_ci'], results_collective_reward_cumulative['collective_R_mean']+results_collective_reward_cumulative['collective_R_ci'], facecolor='lightpink', linewidth=0.04, alpha=1)
   
    plt.title(f'Collective reward (cumulative) in population \n {population_list} ') 
    #plt.gca().set_ylim([0, max(results_collective_reward_cumulative['collective_R_mean'])+1000])
    plt.ylabel(r'Collective reward (cumulative)'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')
    
    plt.savefig(f'{destination_folder}/plots/episode_collective_R_cumul_{population_list}.png', bbox_inches='tight')

    
    

    #do GINI_REWARD - non-cumulative 
    results_gini_reward['gini_R_mean'] = results_gini_reward.mean(axis=1)
    sds = results_gini_reward.std(axis=1)
    results_gini_reward['gini_R_ci'] = 1.96 * sds/np.sqrt(n_runs)

    #plot results - non-cumulative 
    plt.figure(dpi=80, figsize=(15, 6)) 
    plt.plot(results_gini_reward.index[:], results_gini_reward['gini_R_mean'], label=f'{population_list}', color='orange', linewidth=0.05)
    plt.fill_between(results_gini_reward.index[:], results_gini_reward['gini_R_mean']-results_gini_reward['gini_R_ci'], results_gini_reward['gini_R_mean']+results_gini_reward['gini_R_ci'], facecolor='yellow', linewidth=0.04, alpha=1)
   
    plt.title(f'Gini reward (mean) in population \n {population_list} ') 
    plt.gca().set_ylim([0, 1])
    plt.ylabel(r'Gini reward (mean per episode)'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')

    plt.savefig(f'{destination_folder}/plots/episode_gini_R_{population_list}.png', bbox_inches='tight')


    #do MIN_REWARD - non-cumulative 
    results_min_reward['min_R_mean'] = results_min_reward.mean(axis=1)
    sds = results_min_reward.std(axis=1)
    results_min_reward['min_R_ci'] = 1.96 * sds/np.sqrt(n_runs)

    #plot results - non-cumulative 
    plt.figure(dpi=80, figsize=(15, 6)) 
    plt.plot(results_min_reward.index[:], results_min_reward['min_R_mean'], label=f'{population_list}', color='blue', linewidth=0.05)
    plt.fill_between(results_min_reward.index[:], results_min_reward['min_R_mean']-results_min_reward['min_R_ci'], results_min_reward['min_R_mean']+results_min_reward['min_R_ci'], facecolor='lightblue', linewidth=0.04, alpha=1)
   
    plt.title(f'Min reward (mean) in population \n {population_list} ') 
    plt.gca().set_ylim([0, 3])
    plt.ylabel(r'Min reward (mean per episode)'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')

    plt.savefig(f'{destination_folder}/plots/episode_min_R_{population_list}.png', bbox_inches='tight')






def calculate_dil_per_episode(destination_folder, n_runs, population_size, cutoff=[0,15000]):
        '''create dfs for each moral player type, regardless of whether they acted as player1 or player2'''
        colnames = ['run'+str(i+1) for i in range(n_runs)]

        results_runs = pd.DataFrame(columns=colnames)

        path = str(destination_folder+'/history/'+'*.csv')
        
        for run in glob.glob(path): #os.listdir():
            print(f'proecssing run {run}')
            run_name = str(run).replace(str(destination_folder)+'/history/', '')
            if '.csv' in run_name:
                run_name = str(run_name).strip('.csv')

                run_df = pd.read_csv(run)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]

                cutoff_limits = [i*population_size for i in cutoff] #first half of episodes
                counts_selected = pd.DataFrame(run_df[cutoff_limits[0]:cutoff_limits[1]]['idx_p2'].value_counts())
                counts_selected.columns = ['n_times_selected']

                mean_count_selected_this_run = counts_selected.mean() / (cutoff[1]-cutoff[0])

                results_runs[run_name] = mean_count_selected_this_run

        return results_runs



def get_opponents_from_title(long_titles, population_size, title='Selfish', idx=0):
    opponents_temp = long_titles.copy()
    opponents_temp.remove(title)
    #opp_indices = list(range(1, population_size))
    opp_indices = [i for i in list(range(0, population_size)) if i != idx]
    opponents = ["{}{}".format(name, idx) for name, idx in zip(opponents_temp, opp_indices)]
    return opponents


#analyse how often each opponent is selected
def plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range='full'): 
    '''note episode_range uses non-0 indexing, so we need to convert human-readable indices to python-readable'''
    palette = [color_mapping_longtitle[t] for t in long_titles]
    order = list(zip(long_titles, range(population_size)))
    indices_str = [str(i) for i in list(range(population_size))] 
    order = ['_'.join(z) for z in zip(long_titles, indices_str)]
    
    results = pd.DataFrame(index=order)

    if episode_range == 'full':
        episode_idxs = list(range(0,num_iter-1))
        range_label = 'across full episode range'
    elif len(episode_range) == 1: 
        episode_idxs = [i-1 for i in episode_range]
        range_label = f'on episode {episode_idxs[0]+1}'
    elif len(episode_range) > 1:
        episode_idxs = list(range(episode_range[0]-1,episode_range[1]-1))
        range_label = f'on episodes {episode_range[0]+1}-{episode_range[1]+1}'


    for run in range(1,n_runs+1):  
        df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'title_p2', 'idx_p2']]
        small_df = df.loc[ df['episode'].isin(episode_idxs) ]
        small_df['title_idx_p2'] = small_df['title_p2'] + '_' + small_df['idx_p2'].astype(str)

        results[f'run{run}'] = small_df['title_idx_p2'].value_counts()
        #means = results.mean(axis=1)
        #sds = results.std(axis=1)
        #cis = 1.96 * sds/np.sqrt(n_runs)


    chart = sns.barplot(data=results.T, order=order, palette=palette) 
    #chart = sns.countplot(data=results, x=results['run0'], order=order, palette=palette) #sns.countplot(results_df.stack().reset_index(drop=True)) #x=results_df['title_p2'],
    #sns.set(font_scale = 0.5)
    chart.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='small'
        )
    chart.set_title(f'Number of times each player is selected \n {range_label} \n (average across {n_runs} runs)')
    chart.set(ylabel='Number of times selected')

    plt.savefig(f'{destination_folder}/plots/episode_selections_episoderange{episode_range}.png', bbox_inches='tight')




def plot_actions_aggbyplayertype_avgacrossruns(destination_folder, titles, n_runs, num_iter, population_size, title='Selfish'):
    '''explore the actions taken by player types aggregated per episode - what percentage are cooperating? '''

    if False:     
        titles_short = [title.replace('QL','') for title in titles] 
        temp = ''
        for t in titles_short: 
            temp += (t+',')
        population_list = temp.strip(',')
        if len(titles)>6: 
            population_list = destination_folder.replace('results/', '')

    linewidth = 0.005
    
    results_allplayersthistype = pd.DataFrame(index=range(num_iter))
    episodes_column = np.repeat(range(num_iter), population_size)
    indices_this_type = [i for i, x in enumerate(long_titles) if x == title]

    #plot for each player type: 
    for idx in indices_this_type:

        actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
        results_idx = pd.DataFrame(index=range(num_iter)) #actions_idx.index)

        #manually add episode to the actions_df 
#TO DO check that this will work correctly in all cases and with NAs in Selfish player's actions !!! 

        #mask D actions, only keeping Cc so we can the count 
        actions_C = actions_idx.mask(actions_idx.ne('C'))
        actions_C['episode'] = episodes_column
        
        actions_C_episode = actions_C.groupby('episode').count()

        actions_idx['episode'] = episodes_column
        actions_all_episode = actions_idx.groupby('episode').count()

        #calculate % of 100 agents (runs) that cooperate at every step out of the 10000
        results_idx['%_cooperate'] = actions_C_episode.sum(axis='columns') / actions_all_episode.sum(axis='columns') * 100

        results_allplayersthistype[f'{title}_{idx}'] = results_idx['%_cooperate']

    results_allplayersthistype['mean'] = results_allplayersthistype.mean(axis=1)
    results_allplayersthistype['sd'] = results_allplayersthistype.std(axis=1)
    results_allplayersthistype['ci'] = 1.96 * results_allplayersthistype['sd']/np.sqrt(n_runs)

    #plot results 
    plt.figure(dpi=80, figsize=(7, 6)) #3, 2.5
    
    if movingaverage: 
        movingaverage_option = f'\n (moving average, window {movingN})'
        linewidth = 0.5

        movMean = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).mean()
        #movStd = results_allplayersthistype['sd'].rolling(window=movingN,center=True,min_periods=1).std()
        movStd = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).std()
        # get mean +/- 1.96*std/sqrt(N)
        #confIntPos = movMean + 1.96 * movStd / np.sqrt(movingN)
        #confIntNeg = movMean - 1.96 * movStd / np.sqrt(movingN)
        confIntPos = movMean + 1.96 * movStd / np.sqrt(n_runs)
        confIntNeg = movMean - 1.96 * movStd / np.sqrt(n_runs)


        plt.plot(results_allplayersthistype.index[:], movMean, color='darkgreen', linewidth=linewidth)
        plt.fill_between(results_allplayersthistype.index[:], confIntPos, confIntNeg, color='lightgreen', linewidth=linewidth, alpha=0.8) #label=f'{title}_{idx}',    
        
    else: #if plotting all values 
        plt.plot(results_allplayersthistype.index[:], results_allplayersthistype['mean'], color='darkgreen', linewidth=linewidth)
        plt.fill_between(results_allplayersthistype.index[:], results_allplayersthistype['mean']+results_allplayersthistype['ci'], results_allplayersthistype['mean']-results_allplayersthistype['ci'], color='lightgreen', linewidth=linewidth, alpha=0.8) #label=f'{title}_{idx}',    
        
    plt.title(f'% Cooperating (mean over {len(indices_this_type)} {title} players) ') 
    plt.gca().set_ylim([-10, 110])
    plt.ylabel(r'% Cooperating'+ 
                f'\n (mean over '+str(n_runs)+f' runs \n & {len(indices_this_type)} {title} players' + r'+- CI)')
    plt.xlabel('Episode')


    if not os.path.isdir(f'{destination_folder}/plots'):
        os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_actions_agg{title}_avgacrossruns_{movingaverage_option}.png', bbox_inches='tight')


################################################
#### combine different populations on one plot ###
################################################

movingN = 1000 

def manypopulations_plot_cooperation_aggbyplayertype_avgacrossruns(POPULATIONS, num_iter, title='Selfish'):
    '''plot cooperation levels by this speicifc player title across many populationns. 
    color each population by the color of the majority player in it. '''
    episodes_column = np.repeat(range(num_iter), population_size_approx)
    linewidth=1.2 #0.009

    results_manypopulations = pd.DataFrame(index=range(num_iter))
    ci_manypopulations = pd.DataFrame(index=range(num_iter))

    for destination_folder in POPULATIONS:  
        titles = get_titles_for_population(destination_folder)
        population_size = len(titles)
        long_titles = [title_mapping[title] for title in titles]
        indices_this_type = [i for i, x in enumerate(long_titles) if x == title]

        results_allplayersthistype = pd.DataFrame(index=range(num_iter))
        ci_allplayersthistype = pd.DataFrame(index=range(num_iter))

        population = destination_folder.replace('___iter30000_runs20', '')
        print(f'calculating for populaiton {population}')
        #plot for each player type: 
        for idx in indices_this_type:

            actions_idx = pd.read_csv(f'results/{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
            results_idx = pd.DataFrame(index=range(num_iter)) #actions_idx.index)
            ci_idx = pd.DataFrame(index=range(num_iter))

            #manually add episode to the actions_df 
        #TO DO check that this will work correctly in all cases and with NAs in Selfish player's actions !!! 

            #mask D actions, only keeping Cc so we can the count 
            actions_C = actions_idx.mask(actions_idx.ne('C'))
            actions_C['episode'] = episodes_column
                
            actions_C_episode = actions_C.groupby('episode').count()

            actions_idx['episode'] = episodes_column
            actions_all_episode = actions_idx.groupby('episode').count()

            #calculate % of 100 agents (runs) that cooperate at every step out of the 10000
            #results_idx['%_cooperate'] = actions_C_episode.sum(axis='columns') / actions_all_episode.sum(axis='columns') * 100
            percentages_df = actions_C_episode / actions_all_episode * 100
            percentages_df_rolling = percentages_df.rolling(window=movingN,center=True,min_periods=1).mean()

            results_idx['%_cooperate'] = percentages_df_rolling.mean(axis=1) 
            ci_idx['%_cooperate'] = 1.96 * percentages_df_rolling.std(axis=1) / np.sqrt(n_runs_global)

            results_allplayersthistype[f'{title}_{idx}'] = results_idx['%_cooperate']
            ci_allplayersthistype[f'{title}_{idx}'] = ci_idx['%_cooperate']

        #if applicable, average allplayersthistype 
        if len(indices_this_type) > 1: 
            results_allplayersthistype['mean'] = results_allplayersthistype.mean(axis=1)
            ci_allplayersthistype['mean'] = ci_allplayersthistype.mean(axis=1)
        else: 
            results_allplayersthistype['mean'] = results_allplayersthistype#[columns[0]]
            ci_allplayersthistype['mean'] = ci_allplayersthistype#[columns[0]]

        #results_allplayersthistype['mean'] = results_allplayersthistype.mean(axis=1)
        #results_allplayersthistype['sd'] = results_allplayersthistype.std(axis=1)
        #results_allplayersthistype['ci'] = 1.96 * results_allplayersthistype['sd']/np.sqrt(n_runs)


        if False: 
            movingaverage_option = f'\n (moving average, window {movingN})'
            linewidth = 0.8 #overwrite the linewidth with a larger value than above 

            movMean = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).mean()
            #movStd = results_allplayersthistype['sd'].rolling(window=movingN,center=True,min_periods=1).std()
            movStd = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).std()
            # get mean +/- 1.96*std/sqrt(N)
            #confIntPos = movMean + 1.96 * movStd / np.sqrt(movingN)
            #confIntNeg = movMean - 1.96 * movStd / np.sqrt(movingN)
            confInt = 1.96 * movStd / np.sqrt(n_runs)

            results_manypopulations[population] = movMean
            ci_manypopulations[population] = confInt

            #plt.plot(results_allplayersthistype.index[:], movMean, color='darkgreen', linewidth=linewidth)
            #plt.fill_between(results_allplayersthistype.index[:], confIntPos, confIntNeg, color='lightgreen', linewidth=linewidth, alpha=0.8) #label=f'{title}_{idx}',    
            
        #if plotting all values
        movingaverage_option = f'\n (moving avg., window {movingN})'

        results_manypopulations[population] = results_allplayersthistype['mean']
        ci_manypopulations[population] = ci_allplayersthistype['mean']

    print('plotting...')
    #plot results across populations:
    plt.figure(dpi=80, figsize=(6, 6)) #3, 2.5
    sns.set(font_scale=2.5)
    
    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        short_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(short_titles), key = short_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        plt.plot(results_manypopulations.index[:], results_manypopulations[population], linewidth=linewidth, alpha=1, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations.index[:], results_manypopulations[population]+ci_manypopulations[population], results_manypopulations[population]-ci_manypopulations[population], linewidth=linewidth, alpha=0.1, color=color_mapping_longtitle[majority_title])
    #plt.legend(loc='lower center', fontsize="10", bbox_to_anchor=(0.5, -.60))
    # change the line width for the legend
    leg = plt.legend(title='Population', loc='right', bbox_to_anchor=(1.70, 0.5), fontsize="21")
    for line in leg.get_lines():
        line.set_linewidth(5.0)
    title_forplot = title_mapping_forpaper_longtoclean[title] #title_mapping_forpaper_long[title]
    plt.title(f'Cooperation by {title_forplot} player(s) \n across {len(POPULATIONS)} populations ') #% 
    plt.gca().set_ylim([-10, 110])
    plt.ylabel(r'% Cooperating'+ 
                f'\n (mean per episode) {movingaverage_option} \n (mean over 20 runs '+ r'+- CI)')
    plt.xlabel('Episode')

    plt.savefig(f'outcome_plots/manypopulaitons_actions_agg{title}_avgacrossruns_{movingaverage_option}_new.png', bbox_inches='tight')
    

def manypopulations_plot_reward_aggbyplayertype_avgacrossruns(POPULATIONS, titles, n_runs, num_iter, population_size, which='game', cumulative = False, title='Selfish', agg='sum'):
    ''' This plots the non-cumulative reward for 
    which = 'game' or 'intr' '''
    episodes_column = np.repeat(range(num_iter), population_size)

    results_manypopulations = pd.DataFrame(index=range(num_iter))
    ci_manypopulations = pd.DataFrame(index=range(num_iter))

    for destination_folder in POPULATIONS: 

        print('looping over population:')
        titles = get_titles_for_population(destination_folder)
        population_size = len(titles)
        long_titles = [title_mapping[title] for title in titles]
        indices_this_type = [i for i, x in enumerate(long_titles) if x == title]

        results_allplayersthistype = pd.DataFrame(index=range(num_iter))

        population = destination_folder.replace('___iter30000_runs20', '')

        #plot for each player type: 
        for idx in indices_this_type:
            idx_title = long_titles[idx]

            rewards_idx = pd.read_csv(f'results/{destination_folder}/player_rewards/R_{which}_{idx_title}_{idx}.csv', index_col=0).sort_index()
            results_idx = pd.DataFrame(index=range(num_iter)) #actions_idx.index)
    #TO DO check that this will work correctly in all cases and with NAs in Selfish player's actions !!! 
            
            #manually add episode to the actions_df 
            rewards_idx['episode'] = episodes_column
            if agg == 'mean': 
                rewards_idx_episode = rewards_idx.groupby('episode').mean() #calcualte average reward per episode (on every run) 
            elif agg == 'sum': 
                rewards_idx_episode = rewards_idx.groupby('episode').sum() #calcualte total reward per episode (on every run) 

            if cumulative: 
                rewards_idx_episode_cumul = np.cumsum(rewards_idx_episode, axis=0)
                #rewards_idx_episode[f'R_{which}_cumul_mean'] = rewards_idx_episode_cumul.mean(axis=1)
                results_allplayersthistype[f'{title}_{idx}'] = rewards_idx_episode_cumul.mean(axis=1)

            else: #if not using cumulative reward values 
                #rewards_idx_episode[f'R_{which}_mean'] = rewards_idx_episode.mean(axis=1)
                results_allplayersthistype[f'{title}_{idx}'] = rewards_idx_episode.mean(axis=1)

            #results_allplayersthistype[f'{title}_{idx}'] = rewards_idx_episode[f'R_{which}_mean']

                

        results_allplayersthistype['mean'] = results_allplayersthistype.mean(axis=1)
        results_allplayersthistype['sd'] = results_allplayersthistype.std(axis=1)
        results_allplayersthistype['ci'] = 1.96 * results_allplayersthistype['sd']/np.sqrt(n_runs)

        results_manypopulations[population] = results_allplayersthistype['mean']
        ci_manypopulations[population] = results_allplayersthistype['ci']

    print(f'plotting {which} reward...')
    #plot results across populations:
    if cumulative: 
        linewidth=0.4
    else: 
        linewidth=0.01
    plt.figure(dpi=80, figsize=(7, 6)) #3, 2.5
    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        plt.plot(results_manypopulations.index[:], results_manypopulations[population], linewidth=linewidth, alpha=0.9, label=population)
        plt.fill_between(results_manypopulations.index[:], results_manypopulations[population]+ci_manypopulations[population], results_manypopulations[population]-ci_manypopulations[population], color='lightgrey', linewidth=linewidth, alpha=0.8)    
    #plt.legend(loc='lower center', fontsize="10", bbox_to_anchor=(0.5, -.60))
    # change the line width for the legend
    leg = plt.legend(loc='lower center', fontsize="10", bbox_to_anchor=(0.5, -.60))
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    if cumulative: 
        cumul_option = 'Cumulative'
    else: #if cumulative == False 
        cumul_option = '' 
        if agg == 'mean':
            if which == 'game': 
                plt.gca().set_ylim([0, 4])
    plt.title(f'{cumul_option} {which.capitalize()} Reward for {title} players \n across {len(POPULATIONS)} populations') 
    plt.ylabel(f'{cumul_option} {which.capitalize()} Reward'+ 
                f'\n ({agg} per episode)' +
                f'\n (average across {str(n_runs)} runs ' + r'+- CI)') 
    plt.xlabel('Episode')

    if not os.path.isdir('outcome_plots'):
        os.makedirs('outcome_plots')

    plt.savefig(f'outcome_plots/manypopulaitons_{cumul_option}{which}reward_agg{title}_avgacrossruns.png', bbox_inches='tight')
    


movingN = 200 
 
def manypopulations_calculate_cooperation_allplayers_avgacrossruns(POPULATIONS, num_iter):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating?
    % per episode 
    Note this uses the global variable movingN'''

    results_manypopulations = pd.DataFrame(index=range(num_iter))
    ci_manypopulations = pd.DataFrame(index=range(num_iter))

    
    for destination_folder in POPULATIONS: 
        n_runs = len(glob.glob('results/'+destination_folder+'/history/*.csv'))
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'processing {population} with {n_runs} runs')

        result_population = pd.DataFrame(index=range(num_iter)) 

        for run_idx in range(n_runs):
            run_idx += 1
            run_df = pd.read_csv(f'results/{destination_folder}/history/run{run_idx}.csv', index_col=0)[['episode', 'action_player1', 'action_player2']]

            #create boolean variabnle for cooperative selections amde 
            run_df['action_player1_C'] = run_df.apply(lambda row: row['action_player1']==0, axis=1)
            run_df['action_player2_C'] = run_df.apply(lambda row: row['action_player2']==0, axis=1)

            #group by episode 
            run_grouped = run_df.groupby('episode').agg({'action_player1_C': 'sum', 'action_player2_C': 'sum', 'action_player1':'count', 'action_player2':'count'})
            run_grouped['action_C'] = run_grouped['action_player1_C'] + run_grouped['action_player2_C']
            run_grouped['actions_total'] = run_grouped['action_player1'] + run_grouped['action_player2']
            run_grouped['%_C'] = run_grouped['action_C'] * 100 / run_grouped['actions_total']

            
            #EITHER add raw result to output df 
            #result_population[f'run{run_idx}'] = run_grouped['%_C']

            #OR calcualte moving average for each run before storing data 
            rolling_mean_thisrun = run_grouped['%_C'].rolling(window=movingN,center=True,min_periods=1).mean()
            result_population[f'run{run_idx}'] = rolling_mean_thisrun

        #average the rolling means across runs and calcualte CI 
        means = result_population.mean(axis=1)
        sds = result_population.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations[population] = means
        ci_manypopulations[population] = ci

    #save data 
    if not os.path.isdir('outcome_data'):
        os.makedirs('outcome_data')

    if not os.path.isdir('outcome_data/cooperation_allplayers'):
        os.makedirs('outcome_data/cooperation_allplayers')
 
    results_manypopulations.to_csv('outcome_data/cooperation_allplayers/results_manypopulations.csv')
    ci_manypopulations.to_csv('outcome_data/cooperation_allplayers/ci_manypopulations.csv')

def manypopulations_plot_cooperation_allplayers_avgacrossruns(POPULATIONS, num_iter):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating? 
    Plot all populations on one plot.'''
    
    #read in data 
    results_manypopulations = pd.read_csv('outcome_data/cooperation_allplayers/results_manypopulations.csv', index_col=0)
    ci_manypopulations = pd.read_csv('outcome_data/cooperation_allplayers/ci_manypopulations.csv', index_col=0)

    #plot results 
    plt.figure(dpi=80, figsize=(6, 6)) 
    linewidth = 1 #0.1 #0.06
    sns.set(font_scale=2.5)

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        short_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(short_titles), key = short_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'


        if False: 
            movingaverage_option = f'\n (moving average, window {movingN})'
            linewidth = 0.8 #overwrite the linewidth with a larger value than above 

            movMean = results_manypopulations[population].rolling(window=movingN,center=True,min_periods=1).mean()
            #TO DO figure out if I can use the std of the mean or if I need to use the std of the raw data
            movStd = results_manypopulations[population].rolling(window=movingN,center=True,min_periods=1).std()
            # get mean +/- 1.96*std/sqrt(N)
            #confIntPos = movMean + 1.96 * movStd / np.sqrt(movingN)
            #confIntNeg = movMean - 1.96 * movStd / np.sqrt(movingN)
            confInt = 1.96 * movStd / np.sqrt(n_runs_global)

            results_manypopulations[population] = movMean
            ci_manypopulations[population] = confInt

            plt.plot(results_manypopulations.index[:], movMean, linewidth=linewidth, alpha=0.8, label=population_forpaper, color=color_mapping_longtitle[majority_title])
            plt.fill_between(results_manypopulations.index[:], movMean-confInt, movMean+confInt, linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    
        #plotting the moving average from stored data
        movingaverage_option = f'\n (moving avg., window {movingN})'
        #linewidth = 0.8 #overwrite the linewidth with a larger value than above 

        plt.plot(results_manypopulations.index[:], results_manypopulations[population], linewidth=linewidth, alpha=0.8, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations.index[:], results_manypopulations[population]-ci_manypopulations[population], results_manypopulations[population]+ci_manypopulations[population], linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    
    # change the line width for the legend
    leg = plt.legend(title='Population', loc='right', bbox_to_anchor=(1.70, 0.5), fontsize="21")
    for line in leg.get_lines():
        line.set_linewidth(5.0)

    plt.title(f'Cooperation \n across {len(POPULATIONS)} populations ') 
    plt.gca().set_ylim([-5, 105])
    plt.ylabel(r'% Players Cooperating'+ 
                f'\n (mean over 20 runs '+ r'+- CI)' +movingaverage_option) #fontsize=25
    plt.xlabel('Episode')

    if not os.path.isdir('outcome_plots'):
        os.makedirs('outcome_plots')

    plt.savefig('outcome_plots/episode_cooperation_ALLpopulations.pdf', bbox_inches='tight')
    plt.savefig('outcome_plots/episode_cooperation_ALLpopulations.png', bbox_inches='tight')



    if False: #likely saved this from whenI was plotting a previsou version 
        '''plot cooperation levels by this speicifc player title across many populationns. 
        color each population by the color of the majority player in it. '''
        linewidth=0.009

        results_manypopulations = pd.DataFrame(index=range(num_iter))
        ci_manypopulations = pd.DataFrame(index=range(num_iter))

        for destination_folder in POPULATIONS:  

            print('titles in this population:')
            titles = get_titles_for_population(destination_folder)
            population_size = len(titles)
            long_titles = [title_mapping[title] for title in titles]
            indices_this_type = [i for i, x in enumerate(long_titles) if x == title]

            episodes_column = np.repeat(range(num_iter), population_size)


            results_allplayersthistype = pd.DataFrame(index=range(num_iter))

            population = destination_folder.replace('___iter30000_runs20', '')

            #plot for each player type: 
            for idx in indices_this_type:

                actions_idx = pd.read_csv(f'results/{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
                results_idx = pd.DataFrame(index=range(num_iter)) #actions_idx.index)

                #manually add episode to the actions_df 
        #TO DO check that this will work correctly in all cases and with NAs in Selfish player's actions !!! 

                #mask D actions, only keeping Cc so we can the count 
                actions_C = actions_idx.mask(actions_idx.ne('C'))
                actions_C['episode'] = episodes_column
                
                actions_C_episode = actions_C.groupby('episode').count()

                actions_idx['episode'] = episodes_column
                actions_all_episode = actions_idx.groupby('episode').count()

                #calculate % of 100 agents (runs) that cooperate at every step out of the 10000
                results_idx['%_cooperate'] = actions_C_episode.sum(axis='columns') / actions_all_episode.sum(axis='columns') * 100

                results_allplayersthistype[f'{title}_{idx}'] = results_idx['%_cooperate']

            results_allplayersthistype['mean'] = results_allplayersthistype.mean(axis=1)
            results_allplayersthistype['sd'] = results_allplayersthistype.std(axis=1)
            results_allplayersthistype['ci'] = 1.96 * results_allplayersthistype['sd']/np.sqrt(n_runs)


            if movingaverage: 
                movingaverage_option = f'\n (moving average, window {movingN})'
                linewidth = 0.8 #overwrite the linewidth with a larger value than above 

                movMean = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).mean()
                #movStd = results_allplayersthistype['sd'].rolling(window=movingN,center=True,min_periods=1).std()
                movStd = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).std()
                # get mean +/- 1.96*std/sqrt(N)
                #confIntPos = movMean + 1.96 * movStd / np.sqrt(movingN)
                #confIntNeg = movMean - 1.96 * movStd / np.sqrt(movingN)
                confInt = 1.96 * movStd / np.sqrt(n_runs)

                results_manypopulations[population] = movMean
                ci_manypopulations[population] = confInt

                #plt.plot(results_allplayersthistype.index[:], movMean, color='darkgreen', linewidth=linewidth)
                #plt.fill_between(results_allplayersthistype.index[:], confIntPos, confIntNeg, color='lightgreen', linewidth=linewidth, alpha=0.8) #label=f'{title}_{idx}',    
                
            else: #if plotting all values
                movingaverage_option = '' 

                results_manypopulations[population] = results_allplayersthistype['mean']
                ci_manypopulations[population] = results_allplayersthistype['ci']

        print('plotting...')
        #plot results across populations:
        plt.figure(dpi=80, figsize=(7, 6)) #3, 2.5
        for destination_folder in POPULATIONS: 
            population = destination_folder.replace('___iter30000_runs20', '') 
            long_titles = get_titles_for_population(destination_folder)
            majority_title_short = max(set(long_titles), key = long_titles.count)
            majority_title = title_mapping[majority_title_short]
            plt.plot(results_manypopulations.index[:], results_manypopulations[population], linewidth=linewidth, alpha=1, label=population, color=color_mapping_longtitle[majority_title])
            plt.fill_between(results_manypopulations.index[:], results_manypopulations[population]+ci_manypopulations[population], results_manypopulations[population]-ci_manypopulations[population], linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])   #color='lightgrey',  
        #plt.legend(loc='lower center', fontsize="10", bbox_to_anchor=(0.5, -.60))
        # change the line width for the legend
        leg = plt.legend(loc='lower center', fontsize="10", bbox_to_anchor=(0.5, -.60))
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        plt.title(f'% Cooperation by {title} player(s) \n across {len(POPULATIONS)} populations ') 
        plt.gca().set_ylim([-10, 110])
        plt.ylabel(r'% Cooperating'+ 
                    f'\n (mean per episode) {movingaverage_option} \n (mean over {n_runs} runs '+ r'+- CI)')
        plt.xlabel('Episode')

        plt.savefig(f'outcome_plots/manypopulaitons_actions_agg{title}_avgacrossruns_{movingaverage_option}.png', bbox_inches='tight')
        



def manypopulations_calculate_outcomes_allplayers_avgacrossruns(POPULATIONS, num_iter):
    '''For each outcome type (collective reward, collective reward cumulative, ginin reward, min reward):
    calculate outcomes for each population (average across episodes) and store them in a dataframe in a folder outcome_data.'''
    results_manypopulations_collective_reward = pd.DataFrame(index=range(num_iter))
    results_manypopulations_collective_reward_cumulative = pd.DataFrame(index=range(num_iter))
    results_manypopulations_gini_reward = pd.DataFrame(index=range(num_iter))
    results_manypopulations_min_reward = pd.DataFrame(index=range(num_iter))

    ci_manypopulations_collective_reward = pd.DataFrame(index=range(num_iter))
    ci_manypopulations_collective_reward_cumulative = pd.DataFrame(index=range(num_iter))
    ci_manypopulations_gini_reward = pd.DataFrame(index=range(num_iter))
    ci_manypopulations_min_reward = pd.DataFrame(index=range(num_iter))

    #loop over each population and create store the mean and CI across runs for each outcome
    for destination_folder in POPULATIONS: 
        n_runs = len(glob.glob('results/'+destination_folder+'/history/*.csv'))
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'processing {population} with {n_runs} runs')

        result_population_collective_reward = pd.DataFrame(index=range(num_iter)) 
        result_population_collective_reward_cumulative = pd.DataFrame(index=range(num_iter)) 
        result_population_gini_reward = pd.DataFrame(index=range(num_iter)) 
        result_population_min_reward = pd.DataFrame(index=range(num_iter)) 

        for run_idx in range(n_runs):
            #print(f'run {run_idx}')
            run_idx += 1
            run_df = pd.read_csv(f'results/{destination_folder}/history/run{run_idx}.csv', index_col=0)[['episode', 'reward_collective', 'reward_gini', 'reward_min']]

            #group by episode 
            run_grouped = run_df.groupby('episode').agg({'reward_collective': 'sum', 'reward_gini':'mean', 'reward_min':'mean'})

            #EITHER add raw result to output df 
            #result_population_collective_reward[f'run{run_idx}'] = run_grouped['reward_collective']
            #result_population_collective_reward_cumulative[f'run{run_idx}'] = np.cumsum(run_grouped['reward_collective'])
            #result_population_gini_reward[f'run{run_idx}'] = run_grouped['reward_gini']
            #result_population_min_reward[f'run{run_idx}'] = run_grouped['reward_min']

            #OR calcualte moving average for each run before storing data 
            result_population_collective_reward[f'run{run_idx}'] = run_grouped['reward_collective'].rolling(window=movingN,center=True,min_periods=1).mean()
            result_population_collective_reward_cumulative[f'run{run_idx}'] = np.cumsum(run_grouped['reward_collective'])
            result_population_gini_reward[f'run{run_idx}'] = run_grouped['reward_gini'].rolling(window=movingN,center=True,min_periods=1).mean()
            result_population_min_reward[f'run{run_idx}'] = run_grouped['reward_min'].rolling(window=movingN,center=True,min_periods=1).mean()


        #save each outcome (mean & CI) for this population to the appropriate manypopulations df 
        means = result_population_collective_reward.mean(axis=1)
        sds = result_population_collective_reward.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations_collective_reward[population] = means
        ci_manypopulations_collective_reward[population] = ci

        means = result_population_collective_reward_cumulative.mean(axis=1)
        sds = result_population_collective_reward_cumulative.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations_collective_reward_cumulative[population] = means
        ci_manypopulations_collective_reward_cumulative[population] = ci

        means = result_population_gini_reward.mean(axis=1)
        sds = result_population_gini_reward.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations_gini_reward[population] = means
        ci_manypopulations_gini_reward[population] = ci

        means = result_population_min_reward.mean(axis=1)
        sds = result_population_min_reward.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations_min_reward[population] = means
        ci_manypopulations_min_reward[population] = ci

    #store data to disc 
    print('storing data to outcome_data folder')
    if not os.path.isdir('outcome_data'):
        os.makedirs('outcome_data')

    results_manypopulations_collective_reward.to_csv('outcome_data/results_manypopulations_collective_reward.csv')
    results_manypopulations_collective_reward_cumulative.to_csv('outcome_data/results_manypopulations_collective_reward_cumulative.csv')
    results_manypopulations_gini_reward.to_csv('outcome_data/results_manypopulations_gini_reward.csv')
    results_manypopulations_min_reward.to_csv('outcome_data/results_manypopulations_min_reward.csv')

    ci_manypopulations_collective_reward.to_csv('outcome_data/ci_manypopulations_collective_reward.csv')
    ci_manypopulations_collective_reward_cumulative.to_csv('outcome_data/ci_manypopulations_collective_reward_cumulative.csv')
    ci_manypopulations_gini_reward.to_csv('outcome_data/ci_manypopulations_gini_reward.csv')    
    ci_manypopulations_min_reward.to_csv('outcome_data/ci_manypopulations_min_reward.csv')

def manypopulations_plot_outcomes_allplayers_avgacrossruns(POPULATIONS, num_iter):
    '''explore the social outcomes over time - using data stores in outcome_data.  
    Plot all populations on one plot.'''
    

    #read in outcomes data
    results_manypopulations_collective_reward = pd.read_csv('outcome_data/results_manypopulations_collective_reward.csv', index_col=0)
    results_manypopulations_collective_reward_cumulative = pd.read_csv('outcome_data/results_manypopulations_collective_reward_cumulative.csv', index_col=0)
    results_manypopulations_gini_reward = pd.read_csv('outcome_data/results_manypopulations_gini_reward.csv', index_col=0)
    results_manypopulations_min_reward = pd.read_csv('outcome_data/results_manypopulations_min_reward.csv', index_col=0)

    ci_manypopulations_collective_reward = pd.read_csv('outcome_data/ci_manypopulations_collective_reward.csv', index_col=0)
    ci_manypopulations_collective_reward_cumulative = pd.read_csv('outcome_data/ci_manypopulations_collective_reward_cumulative.csv', index_col=0)
    ci_manypopulations_gini_reward = pd.read_csv('outcome_data/ci_manypopulations_gini_reward.csv', index_col=0)    
    ci_manypopulations_min_reward = pd.read_csv('outcome_data/ci_manypopulations_min_reward.csv', index_col=0)



    #now plot each outcome in turn 
    figsize = (6, 6)
    linewidth = 1 #0.06
    sns.set(font_scale=2.5)

    #do COLLECTIVE_REWARD first - non-cumulative 
    print('plotting collective reward - non-cumulative')
    plt.figure(dpi=80, figsize=figsize) 

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        long_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(long_titles), key = long_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        if False: 
            movingaverage_option = f'\n (moving average, window {movingN})'
            linewidth = 0.5

            movMean = results_manypopulations_collective_reward[population].rolling(window=movingN,center=True,min_periods=1).mean()
            #movStd = results_allplayersthistype['sd'].rolling(window=movingN,center=True,min_periods=1).std()
            movStd = results_manypopulations_collective_reward[population].rolling(window=movingN,center=True,min_periods=1).std()
            #TO DO understand if we can use ci_manypopulations_collective_reward here in calcualting the moving CI 
            # get mean +/- 1.96*std/sqrt(N)
            #confIntPos = movMean + 1.96 * movStd / np.sqrt(movingN)
            #confIntNeg = movMean - 1.96 * movStd / np.sqrt(movingN)
            confIntPos = movMean + 1.96 * movStd / np.sqrt(n_runs_global)
            confIntNeg = movMean - 1.96 * movStd / np.sqrt(n_runs_global)


            plt.plot(results_manypopulations_collective_reward.index[:], movMean, linewidth=linewidth, alpha=1, label=population_forpaper, color=color_mapping_longtitle[majority_title])
            plt.fill_between(results_manypopulations_collective_reward.index[:], confIntPos, confIntNeg, 
                            linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  

        movingaverage_option = f'\n (moving avg., window {movingN})'
        plt.plot(results_manypopulations_collective_reward.index[:], results_manypopulations_collective_reward[population], linewidth=linewidth, alpha=0.7, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations_collective_reward.index[:], 
                            results_manypopulations_collective_reward[population]-ci_manypopulations_collective_reward[population], 
                            results_manypopulations_collective_reward[population]+ci_manypopulations_collective_reward[population], 
                            linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  

    population_size_approx = 16

    #leg = plt.legend(loc='lower center', fontsize="12", bbox_to_anchor=(0.5, -.70))
    #for line in leg.get_lines():
    #    line.set_linewidth(4.0)
    plt.title(f'Collective reward \n across {len(POPULATIONS)} populations') 
    plt.gca().set_ylim([2*population_size_approx, 6*population_size_approx])
    plt.ylabel(f'Collective reward \n (sum per episode) \n (mean across 20 runs' + r' +- CI)' + movingaverage_option) #, fontsize=25
    plt.xlabel('Episode')

    if not os.path.isdir('outcome_plots'):
        os.makedirs('outcome_plots')

    plt.savefig(f'outcome_plots/episode_collective_R_ALLpopulations_{movingaverage_option}.pdf', bbox_inches='tight')
    plt.savefig(f'outcome_plots/episode_collective_R_ALLpopulations_{movingaverage_option}.png', bbox_inches='tight')



    #plot COLLECTIVE_REWARD cumulative 
    print('plotting collective reward - cumulative')
    plt.figure(dpi=80, figsize=figsize) 
    #linewidth = 1

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        long_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(long_titles), key = long_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        plt.plot(results_manypopulations_collective_reward_cumulative.index[:], results_manypopulations_collective_reward_cumulative[population], linewidth=linewidth, alpha=1, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations_collective_reward_cumulative.index[:], 
                         results_manypopulations_collective_reward_cumulative[population]-ci_manypopulations_collective_reward_cumulative[population], 
                         results_manypopulations_collective_reward_cumulative[population]+ci_manypopulations_collective_reward_cumulative[population], 
                         linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    

    leg = plt.legend(title='Population', loc='right', bbox_to_anchor=(1.70, 0.5), fontsize="21")
    for line in leg.get_lines():
        line.set_linewidth(4.5)
    plt.title(f'Collective reward, cumulative \n across {len(POPULATIONS)} populations') 
    plt.ylabel(f'Collective reward \n (sum per episode), \n cumulative'+ f'\n (mean across 20 runs' + r' +- CI)')
    plt.xlabel('Episode')

    plt.savefig('outcome_plots/episode_collective_R_cumulative__ALLpopulations.pdf', bbox_inches='tight')


    #plot GINI_REWARD 
    print('plotting gini reward')
    plt.figure(dpi=80, figsize=figsize) 
    #linewidth = 0.06

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        long_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(long_titles), key = long_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        if False: 
            movingaverage_option = f'\n (moving average, window {movingN})'
            linewidth = 0.5

            movMean = results_manypopulations_gini_reward[population].rolling(window=movingN,center=True,min_periods=1).mean()
            movStd = results_manypopulations_gini_reward[population].rolling(window=movingN,center=True,min_periods=1).std()
            #TO DO understand if we can use ci_manypopulations_collective_reward here in calcualting the moving CI 
            confIntPos = movMean + 1.96 * movStd / np.sqrt(n_runs_global)
            confIntNeg = movMean - 1.96 * movStd / np.sqrt(n_runs_global)


            plt.plot(results_manypopulations_gini_reward.index[:], movMean, linewidth=linewidth, alpha=1, label=population_forpaper, color=color_mapping_longtitle[majority_title])
            plt.fill_between(results_manypopulations_gini_reward.index[:], confIntPos, confIntNeg, 
                            linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
        else: 
            movingaverage_option = f'\n (moving avg., window {movingN})'
            plt.plot(results_manypopulations_gini_reward.index[:], results_manypopulations_gini_reward[population], linewidth=linewidth, alpha=0.7, label=population_forpaper, color=color_mapping_longtitle[majority_title])
            plt.fill_between(results_manypopulations_gini_reward.index[:], 
                         results_manypopulations_gini_reward[population]-ci_manypopulations_gini_reward[population], 
                         results_manypopulations_gini_reward[population]+ci_manypopulations_gini_reward[population], 
                         linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    

    #leg = plt.legend(loc='lower center', fontsize="12", bbox_to_anchor=(0.5, -.70))
    #for line in leg.get_lines():
    #    line.set_linewidth(4.0)
    plt.title(f'Gini reward \n across {len(POPULATIONS)} populations') 
    plt.gca().set_ylim([0, 1])
    plt.ylabel(f'Gini reward \n (mean per episode) \n (mean across 20 runs' + r' +- CI)' + movingaverage_option)
    plt.xlabel('Episode')

    plt.savefig(f'outcome_plots/episode_gini_R_ALLpopulations_{movingaverage_option}.pdf', bbox_inches='tight')
    plt.savefig(f'outcome_plots/episode_gini_R_ALLpopulations_{movingaverage_option}.png', bbox_inches='tight')


    #plot MIN_REWARD 
    print('plotting min reward')
    plt.figure(dpi=80, figsize=figsize) 
    #linewidth = 0.06

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        long_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(long_titles), key = long_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        if False: 
            movingaverage_option = f'\n (moving average, window {movingN})'
            linewidth = 0.5

            movMean = results_manypopulations_min_reward[population].rolling(window=movingN,center=True,min_periods=1).mean()
            movStd = results_manypopulations_min_reward[population].rolling(window=movingN,center=True,min_periods=1).std()
            #TO DO understand if we can use ci_manypopulations_collective_reward here in calcualting the moving CI 
            confIntPos = movMean + 1.96 * movStd / np.sqrt(n_runs_global)
            confIntNeg = movMean - 1.96 * movStd / np.sqrt(n_runs_global)


            plt.plot(results_manypopulations_min_reward.index[:], movMean, linewidth=linewidth, alpha=1, label=population_forpaper, color=color_mapping_longtitle[majority_title])
            plt.fill_between(results_manypopulations_min_reward.index[:], confIntPos, confIntNeg, 
                            linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
        else: 
            movingaverage_option = f'\n (moving avg., window {movingN})'
            plt.plot(results_manypopulations_min_reward.index[:], results_manypopulations_min_reward[population], linewidth=linewidth, alpha=0.7, label=population_forpaper, color=color_mapping_longtitle[majority_title])
            plt.fill_between(results_manypopulations_min_reward.index[:], 
                         results_manypopulations_min_reward[population]-ci_manypopulations_min_reward[population], 
                         results_manypopulations_min_reward[population]+ci_manypopulations_min_reward[population], 
                         linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    

    leg = plt.legend(title='Population', loc='right', bbox_to_anchor=(1.70, 0.5), fontsize="21")
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.title(f'Min reward \n across {len(POPULATIONS)} populations') 
    plt.gca().set_ylim([0, 3])
    plt.ylabel(f'Min reward \n (mean per episode) \n (mean across 20 runs' + r' +- CI)' + movingaverage_option)
    plt.xlabel('Episode')

    plt.savefig(f'outcome_plots/episode_min_R_ALLpopulations_{movingaverage_option}.pdf', bbox_inches='tight')
    plt.savefig(f'outcome_plots/episode_min_R_ALLpopulations_{movingaverage_option}.png', bbox_inches='tight')





def manypopulations_plot_heatmap_reward(POPULATIONS, num_iter, reward_type): 

    figsize=(8, 6.5)
    player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    print('NOTE assuming all player types are possible in population!!')
    player_types_long = [title_mapping[title] for title in player_types]
    #populations = [f'majority {title}' for title in player_types_long]
    player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]
    populations = [f'majority-{title}' for title in player_types_forpaper]

    matrix_rewards = pd.DataFrame(columns=populations, index=player_types_forpaper)

    for destination_folder in POPULATIONS: 

        titles = get_titles_for_population(destination_folder)

        population_size = len(titles)
        destination_folder = 'results/' + destination_folder

        long_titles = [title_mapping[title] for title in titles]

#NOTE this assumes only one player is the majority player in whole population!! 
        multiple_players_same_type_rewards = []
        majority_title = max(set(long_titles), key=long_titles.count)
        majority_title_forpaper = title_mapping_forpaper_long[majority_title]

        for idx in range(population_size):
            idx_title = long_titles[idx]
            idx_title_forpaper = title_mapping_forpaper_long[idx_title]
            #read in cumulative reward values and forward-fill missing iterations with last available value 
            reward_df = pd.read_csv(f'{destination_folder}/player_rewards/Rcumul_{reward_type}_{idx_title}_{idx}.csv', index_col=0).ffill()
            last_avail_reward = reward_df.mean(axis=1).iloc[-1]

            if long_titles.count(idx_title) > 1: #if this player occurs in the population more than once
                if not majority_title == idx_title:
                    print ('ERROR! there is more than one majority player type')
                    break 
                else: 
                    majority_title = idx_title
                multiple_players_same_type_rewards.append(last_avail_reward.astype(float))
            else: #if only one player of this type exists in the population 
                matrix_rewards.loc[idx_title_forpaper, f'majority-{majority_title_forpaper}'] = last_avail_reward.astype(float)

        #fill in value for the majority player - average across multiple players 
        multiple_players_avg_reward = mean(multiple_players_same_type_rewards)
        matrix_rewards.loc[majority_title_forpaper, f'majority-{majority_title_forpaper}'] = multiple_players_avg_reward.astype(float)

    #if plotting intrinsic reward_type, drop the Selfish player 
    if reward_type == 'intr':
        matrix_rewards.drop('S', inplace=True)
        player_types_forpaper.remove('S')

        #normalise intr rewards to be between the min and max value for eahc type of player 
        matrix_rewards_T = matrix_rewards.T
        #normalized_df = (matrix_rewards_T-matrix_rewards_T.min()) / (matrix_rewards_T.max()-matrix_rewards_T.min())
        #normalized_df = (matrix_rewards_T-0) / (matrix_rewards_T.max()-0)
        normalized_df = pd.DataFrame(index=matrix_rewards_T.index, columns=matrix_rewards_T.columns)
        for column in matrix_rewards_T.columns:
            if column in ['De', 'aUt']:
                positive_column = matrix_rewards_T[column] - min(matrix_rewards_T[column])
                #normalized_df[column] = (positive_column - 0) / (positive_column.max() - 0)
                normalized_df[column] = (positive_column - positive_column.min()) / (positive_column.max() - positive_column.min())

            elif column in ['Ut', 'V-Eq', 'V-Ki', 'mDe', 'V-In', 'V-Ag']:
                #normalized_df[column] = (matrix_rewards_T[column] - 0) / (matrix_rewards_T[column].max() - 0)
                normalized_df[column] = (matrix_rewards_T[column] - matrix_rewards_T[column].min()) / (matrix_rewards_T[column].max() - matrix_rewards_T[column].min())

        matrix_rewards_toplot = normalized_df.T
        #normalization_option = ' 0-max normalized'
        normalization_option = ' min-max normalized'

        vmax = 1
        cmap = 'pink_r' #'cividis_r' #'plasma_r' #'viridis_r' #magma_r
        reward_type = 'Intrinsic'

    else: #if plotting game reward
        vmax = matrix_rewards.max(axis=None).max()
        normalization_option = ''
        matrix_rewards_toplot = matrix_rewards
        cmap = 'bone_r' #'winter_r' #'plasma_r' #'viridis_r'
    
    for label in matrix_rewards_toplot.columns:
        matrix_rewards_toplot[label] = matrix_rewards_toplot[label].astype(float)

    plt.figure(figsize=figsize)
    sns.set(font_scale=2.8)
    s = sns.heatmap(matrix_rewards_toplot, cmap=cmap, xticklabels=populations, yticklabels=player_types_forpaper,
                    annot=False, linecolor='black', linewidths=0, square=True, vmin = 0, vmax = vmax)
    s.set(xlabel='Population', ylabel='Player Type',title=f'{reward_type.capitalize()} Rewards \n (accumulated over all episodes) \n {normalization_option}')
    #manually set limits to top and bottom of heatmap do not get cut off 
    bottom, top = s.get_ylim()
    s.set_ylim(bottom+0.5, top-0.5)

    plt.savefig(f'outcome_plots/matrix_{reward_type}_rewards.pdf', bbox_inches='tight')
    matrix_rewards_toplot.to_csv(f'outcome_data/matrix_{reward_type}_rewards_{normalization_option}.csv')




def calculate_heatmap_selections(destination_folder, num_iter, num_episodes, episode_idxs):
    titles = get_titles_for_population(destination_folder)

    population_size = len(titles)
    destination_folder = 'results/' + destination_folder

    long_titles = [title_mapping[title] for title in titles]
    #palette = [color_mapping_longtitle[t] for t in long_titles]

    #if using title AND idx 
    #indices_str = [str(i) for i in list(range(population_size))] 
    #order = ['_'.join(z) for z in zip(long_titles, indices_str)]
    #results_allruns = pd.DataFrame(index=order) #long_titles)

    results_allruns = pd.DataFrame(index=set(long_titles))


#NOTE this assumes only one player is the majority player in whole population!! 
    #multiple_players_same_type_rewards = []
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]


    for run in range(1,n_runs_global+1):
        #FROM EARLIER FUNCTION 
        df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'title_p2', 'idx_p2']]
        small_df = df.loc[ df['episode'].isin(episode_idxs) ]
            
        results_allruns[f'run{run}'] = small_df['title_p2'].value_counts()

        #if using title AND idx
        #small_df['title_idx_p2'] = small_df['title_p2'] + '_' + small_df['idx_p2'].astype(str)
        #results_allruns[f'run{run}'] = small_df['title_idx_p2'].value_counts()


    #now aggregate across runs 
    results_allruns_percenttimes_selected = results_allruns.mean(axis=1) / (population_size*num_episodes)
    ci_allruns_percenttimes_selected = ( 1.96 * results_allruns.std(axis=1)/np.sqrt(n_runs_global) ) / (population_size*num_episodes)

    results_allruns_percenttimes_selected.index = [title_mapping_forpaper_long[i] for i in results_allruns_percenttimes_selected.index]
    ci_allruns_percenttimes_selected.index = [title_mapping_forpaper_long[i] for i in ci_allruns_percenttimes_selected.index]

    return majority_title, results_allruns_percenttimes_selected, ci_allruns_percenttimes_selected

def manypopulations_calculate_heatmap_selections(POPULATIONS, num_iter, episode_range=[29900, 30000]): #TO FINISH !!! 

    player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    print('NOTE assuming all player types are possible in population!!')
    player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]

    populations = [f'majority-{title}' for title in player_types_forpaper]

    matrix_selections = pd.DataFrame(columns=populations, index=player_types_forpaper)
    ci_selections = pd.DataFrame(columns=populations, index=player_types_forpaper)

    if episode_range == 'full':
        episode_idxs = list(range(0,num_iter))
        range_label = 'across full episode range'
    elif len(episode_range) == 1: 
        episode_idxs = [i-1 for i in episode_range]
        range_label = f'on episode {episode_idxs[0]}'
    elif len(episode_range) > 1:
        episode_idxs = list(range(episode_range[0],episode_range[1]))
        range_label = f'on episodes {episode_range[0]}-{episode_range[1]}'
        num_episodes = episode_range[1] - episode_range[0] 



    for destination_folder in POPULATIONS: 

        majority_title, results_allruns_percenttimes_selected, ci_allruns_percenttimes_selected = calculate_heatmap_selections(destination_folder, num_iter, num_episodes, episode_idxs) 
    
        #append to results dataframe 
        matrix_selections[f'majority-{majority_title}'] = results_allruns_percenttimes_selected.astype(float)
        ci_selections[f'majority-{majority_title}'] = ci_allruns_percenttimes_selected.astype(float)


    #save selections matrix to csv
    matrix_selections.to_csv(f'outcome_data/matrix_selections_episodes{episode_range}.csv')
    ci_selections.to_csv(f'outcome_data/ci_selections_episodes{episode_range}.csv')

#def plot_heatmap_selections():


def manypopulations_plot_heatmap_selections(POPULATIONS, num_iter, episode_range=[29900, 30000]):
    '''plot heatmap and more detailed bar plot of number of times each player type is selected on the final 100 episdoes in each populaiton'''
    figsize=(6, 6)

    matrix_selections = pd.read_csv(f'outcome_data/matrix_selections_episodes{episode_range}.csv', index_col=0)
    matrix_selections_percent = matrix_selections*100 

    ci_selections = pd.read_csv(f'outcome_data/ci_selections_episodes{episode_range}.csv', index_col=0)
    ci_selections_percent = ci_selections*100 

    player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    player_types_long = [title_mapping[title] for title in player_types]

    print('NOTE assuming all player types are possible in population!!')
    player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]
    populations = [f'majority-{title}' for title in player_types_forpaper]


    #plot selections matrix
    plt.figure(figsize=figsize)
    sns.set(font_scale=2)
    s = sns.heatmap(matrix_selections_percent, cmap='gist_yarg', 
                    #viridis, seismic, coolwarm, cool, hot, PiYG, summer, PuOr, PRGn, Blues, Greens, Purples, PuBuGn, YlGnBu, BuGn, gist_yarg, RdPu, PuRd
                     xticklabels=populations, yticklabels=player_types_forpaper, 
                    annot=False, linecolor='black', linewidths=0, square=True, vmin = 0, vmax = 100)
    s.set(xlabel='Population', ylabel='Player Type',
          title=f'Popularity of each player type \n (% times selected) \n (over last 100 episodes)')
    #manually set limits to top and bottom of heatmap do not get cut off 
    bottom, top = s.get_ylim()
    s.set_ylim(bottom+0.5, top-0.5)

    plt.savefig('outcome_plots/matrix_selections.pdf', bbox_inches='tight')


#def manypopulations_plot_bar_selections(POPULATIONS, num_iter, episode_range=[29900, 30000]):
#TO DO add calculation where we treat each player separately 

    #plot selections as bar chart 
    palette = [color_mapping_longtitle[t] for t in player_types_long]

    sns.set(font_scale=2)
    #s = matrix_selections_percent.T.plot(kind='bar', stacked=False, color=palette, width=0.8, figsize=(15, 6), edgecolor='grey')
    s = matrix_selections_percent.T.plot.bar(stacked=False, yerr=ci_selections_percent.T, color=palette, width=0.8, figsize=(15, 6), edgecolor='grey')
    #s = plt.errorbar(y=matrix_selections_percent.T, yerr=ci_selections_percent.T)
    # horizontal line indicating the threshold
    s.set(xlabel='Population', ylabel='% times selected \n (mean over last 100 episodes)',
          title=f'Popularity of each player type \n (over last 100 episodes)')
    leg = plt.legend(title='Player', loc='right', bbox_to_anchor=(1.2, 0.5)) #, fontsize="21"
    s.axhline(y=50, color='grey', linestyle='--') #label = '50% threshold'
    s.annotate('majority player constitutes \n '+r'50% of populaiton', xy=(7, 53), ha="center", va="top", fontsize=17)


    plt.savefig('outcome_plots/bar_selections.pdf', bbox_inches='tight')



def manypopulations_calculate_network_selections(POPULATIONS, num_iter, episode_range=[0, 30000]):
    '''create a matrix of when each player type selects each other player type - count number of iterations across episode_range. 
    Then average across all runs '''

    player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    print('NOTE assuming all player types are possible in population!!')
    player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]

    populations = [f'majority {title}' for title in player_types_forpaper]

    if episode_range == 'full':
        episode_idxs = list(range(0,num_iter))
        range_label = 'across full episode range'
    elif len(episode_range) == 1: 
        episode_idxs = [i-1 for i in episode_range]
        range_label = f'on episode {episode_idxs[0]}'
    elif len(episode_range) > 1:
        episode_idxs = list(range(episode_range[0],episode_range[1]))
        range_label = f'on episodes {episode_range[0]}-{episode_range[1]}'
        num_episodes = episode_range[1] - episode_range[0] 

    for destination_folder in POPULATIONS:

        titles = get_titles_for_population(destination_folder)
        population_size = len(titles)
        destination_folder = 'results/' + destination_folder
        long_titles = [title_mapping[title] for title in titles]
        print('assuming only one player is the majority player in whole population!!')
        majority_title_long = max(set(long_titles), key=long_titles.count)
        majority_title = title_mapping_forpaper_long[majority_title_long]
        population_tile_forpaper = f'majority {majority_title}'

        #loop over eveyr run 
        #network_selections = pd.DataFrame(columns=player_types_forpaper, index=player_types_forpaper)
        network_selections_everyrun = dict.fromkeys(list(range(1, n_runs_global+1)))

        for run in range(1,n_runs_global+1):
            #FROM EARLIER FUNCTION 
            df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'title_p1', 'title_p2']]
            small_df = df.loc[ df['episode'].isin(episode_idxs) ]
            small_df.drop('episode', axis='columns', inplace=True)

            small_df["value"]=1
            #pd.pivot_table(small_df, values="value", index=["title_p1"], columns="title_p2", fill_value=0) 
            #small_df.pivot_table(index=["title_p1"], columns="title_p2", aggfunc=lambda x: 1, fill_value=0)
            network_selections_matrix = pd.crosstab(small_df['title_p1'], small_df['title_p2'].fillna(0))
            network_selections_matrix.rename(index=title_mapping_forpaper_long, columns=title_mapping_forpaper_long, inplace=True)

            network_selections_everyrun[run] = network_selections_matrix
            
        #average numbers across all runs
        network_selections_mean = pd.concat(network_selections_everyrun.values()).reset_index().groupby("title_p1").mean()

        #store results in this population's results folder 
        network_selections_mean.to_csv(f'{destination_folder}/network_selections_mean_episodes{episode_range}.csv')



def plot_network_selections(destination_folder, num_iter, episode_range=[0, 30000]): 
    '''plot selections matrix and network graph per player TYPE (aggregarte multiple players of the same time 
    - will be skewed by one player beong predominant)'''
    titles = get_titles_for_population(destination_folder.replace('results/',''))
    if not 'results' in destination_folder: 
        destination_folder = 'results/' + destination_folder
    long_titles = [title_mapping[title] for title in titles]
    print('assuming only one player is the majority player in whole population!!')
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]
    population_tile_forpaper = f'majority-{majority_title}'

    player_types_ordered = ['S', 'Ut', 'De', 'V-Eq', 'V-Ki', 'aUt', 'mDe', 'V-In', 'V-Ag']


    network_selections_mean = pd.read_csv(f'{destination_folder}/network_selections_mean_episodes{episode_range}.csv', index_col=0)
    #add a threshold so the network is not too populated - only keep edges with weights > 10000
    network_selections_mean = network_selections_mean.mask(network_selections_mean < 10000)
    #network_selections_mean.mask(network_selections_mean < network_selections_mean.quantile())

    #change order in the index 
    network_selections_mean_forplotting = network_selections_mean.reindex(index=player_types_ordered, columns=player_types_ordered)
    
    

    # 1 - plot selections matrix as heatmap 
    plt.clf()
    plt.figure(figsize=(6,6))
    sns.set(font_scale=2)
    s = sns.heatmap(network_selections_mean_forplotting, cmap='gist_yarg', 
                    #viridis, seismic, coolwarm, cool, hot, PiYG, summer, PuOr, PRGn, Blues, Greens, Purples, PuBuGn, YlGnBu, BuGn, gist_yarg, RdPu, PuRd
                    xticklabels=player_types_ordered, yticklabels=player_types_ordered, 
                    annot=False, linecolor='black', linewidths=0, square=True, 
                    vmin = 0, vmax = network_selections_mean_forplotting.max(axis=None).max())
    s.set(ylabel='Player 1 (selectOR) Type', xlabel='Player 2 (selectED) Type',
          title=f'Selection dynamics across player types \n in population {population_tile_forpaper} \n (# times selected) \n (over episodes {episode_range[0]}-{episode_range[1]})')
    #manually set limits to top and bottom of heatmap do not get cut off 
    bottom, top = s.get_ylim()
    s.set_ylim(bottom+0.5, top-0.5)

    plt.savefig(f'outcome_plots/matrix_selections_byplayertype_popul{population_tile_forpaper}.pdf', bbox_inches='tight')


    # 2 - plot selctions as network plot, circular layout 
    plt.clf()
    plt.figure(figsize=(4,3))
    temp = network_selections_mean_forplotting.replace(0, np.nan).astype('float')
    #min-max normalise 
    network_selections_mean_fornetworkplotting = temp.sub(temp.min().min()).div((temp.max().max()-temp.min().min()))
    network_selections_mean_fornetworkplotting = network_selections_mean_fornetworkplotting.mul(10)

    G=nx.from_pandas_adjacency(network_selections_mean_fornetworkplotting.astype('float'), create_using=nx.DiGraph())
    #G = nx.DiGraph(network_selections_mean_fornetworkplotting.astype('float'))

    widths = nx.get_edge_attributes(G, 'weight')
    node_colors = [color_mapping_longtitle[title_mapping_forpaper_shorttolong[node]] for node in G.nodes()]
    #elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 10000]
    #esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 10000]

    edges = {k: v for k, v in widths.items() if not isnan(v)}
    #pos = nx.spring_layout(G, k=5.95, seed=7)  # positions for all nodes - seed for reproducibility
    pos = nx.circular_layout(G) #, scale=5, center=('S', 'DE')  # positions for all nodes - seed for reproducibility
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=list(edges.keys()), width=list(edges.values()), alpha=0.5, 
                           edge_color="black", style="solid", arrows=True, arrowstyle='-|>', connectionstyle="arc3,rad=0.1", node_size=500)
    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="serif")
    plt.title(f'Selections \n in population {population_tile_forpaper}')
    plt.savefig(f'outcome_plots/network_selections_byplayertype_popul{population_tile_forpaper}.pdf', bbox_inches='tight')

#for changing the size of self-loops in network plots 
#https://stackoverflow.com/questions/74350464/how-to-better-visualize-networkx-self-loop-plot


def calculate_network_selection_bipartite(destination_folder, episode_idxs, episode_range, num_episodes, reorder_index=False):

    titles = get_titles_for_population(destination_folder)
    population_size = len(titles)
    destination_folder = 'results/' + destination_folder
    long_titles = [title_mapping[title] for title in titles]
    print('for the chart label, assuming only one player is the majority player in whole population!!')
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]
    population_tile_forpaper = f'majority {majority_title}'
    print(f'running for populaiton {population_tile_forpaper}')

    #loop over eveyr run 
    #network_selections = pd.DataFrame(columns=player_types_forpaper, index=player_types_forpaper)
    network_selections_everyrun = dict.fromkeys(list(range(1, n_runs_global+1)))

    for run in range(1,n_runs_global+1):
        #FROM EARLIER FUNCTION 
        df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'idx_p1', 'title_p1', 'idx_p2', 'title_p2']]
        small_df = df.loc[ df['episode'].isin(episode_idxs) ]
        small_df.drop('episode', axis='columns', inplace=True)
        #replace long player title wiht short 
        small_df.replace({'title_p1': title_mapping_forpaper_long, 'title_p2': title_mapping_forpaper_long}, inplace=True)
        
        #reorder index so prosocial players appear first! 
        if reorder_index:
            relabel_dict = relabel_player_index(long_titles)
            small_df['idx_p1'] = small_df['idx_p1'].map(relabel_dict)
            small_df['idx_p2'] = small_df['idx_p2'].map(relabel_dict)

        #concatenate type and index for bipartite graph 
        small_df['title_idx_p1'] = small_df['idx_p1'].astype(str) + '_' + small_df['title_p1']
        small_df['title_idx_p2'] = small_df['idx_p2'].astype(str) + '_' + small_df['title_p2']

        #create unique ordered index for this population 
        labels_df = small_df[['idx_p1','title_p1']].value_counts()
        labels = ['{}_{}'.format(i, j) for i, j in labels_df.index]

        small_df["value"]=1
        network_selections_matrix_temp = pd.crosstab(small_df['title_idx_p1'], small_df['title_idx_p2'].fillna(0))
        #add columns for all labels, even those that were not present in the selections at all
        network_selections_matrix = pd.DataFrame(index=labels, columns=labels)
        #reorder columns and rows so they are in fixed order
        #network_selections_matrix = network_selections_matrix[labels]
        network_selections_matrix = network_selections_matrix.reindex(labels)
        #add crosstab values into results df with all possible columns 
        network_selections_matrix[network_selections_matrix_temp.columns] = network_selections_matrix_temp

        #store results for this run in big df 
        network_selections_everyrun[run] = network_selections_matrix
            
    #average numbers across all runs
    network_selections_mean = pd.concat(network_selections_everyrun.values()).reset_index().groupby("index").mean()
    #reorder index and columns in result 
    network_selections_mean_ordered = network_selections_mean.reindex(index=labels, columns=labels)
    #store results in this population's results folder 
    network_selections_mean_ordered.to_csv(f'{destination_folder}/network_selections_bipartite_mean_episodes{episode_range}.csv')


def manypopulations_calculate_network_selection_bipartite(POPULATIONS, num_iter, episode_range=[0, 30000]):
    '''create a matrix of when each player type selects each other player type - count number of iterations across episode_range. 
    Then average across all runs '''

    #player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    #print('NOTE assuming all player types are possible in population!!')
    #player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]

    #populations = [f'majority {title}' for title in player_types_forpaper]

    if episode_range == 'full':
        episode_idxs = list(range(0,num_iter))
        #range_label = 'across full episode range'
    elif len(episode_range) == 1: 
        episode_idxs = [i-1 for i in episode_range]
        #range_label = f'on episode {episode_idxs[0]}'
    elif len(episode_range) > 1:
        episode_idxs = list(range(episode_range[0],episode_range[1]))
        #range_label = f'on episodes {episode_range[0]}-{episode_range[1]}'
        num_episodes = episode_range[1] - episode_range[0] 

    for destination_folder in POPULATIONS:
        calculate_network_selection_bipartite(destination_folder, episode_idxs, episode_range, num_episodes, reorder_index=True)


def plot_network_selections_bipartite(destination_folder, num_iter, episode_range=[0, 30000], threshold=False, normalise=True, one_run=False): 
    '''plot selections matrix and network graph per INDIVIDUAL PLAYER (aggregarte multiple players of the same time 
    - will be skewed by one player beong predominant)'''

    titles = get_titles_for_population(destination_folder.replace('results/',''))
    if not 'results' in destination_folder: 
        destination_folder = 'results/' + destination_folder
    long_titles = [title_mapping[title] for title in titles]
    print('assuming only one player is the majority player in whole population!!')
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]
    population_tile_forpaper = f'majority-{majority_title}'

    #player_types_ordered = ['S', 'Ut', 'De', 'V-Eq', 'V-Ki', 'aUt', 'mDe', 'V-In', 'V-Ag']
    if one_run:
        run_option = f'_run{one_run}'
    else:
        run_option=''

    network_selections_mean = pd.read_csv(f'{destination_folder}/network_selections_bipartite_mean_episodes{episode_range}{run_option}.csv', index_col=0)

    #if specified, add a threshold so the network is not too populated - only keep edges with weights > 10000
    if threshold: 
        if type(threshold) == bool: #use quantile if no numeric threshold specified 
            network_selections_mean = network_selections_mean.mask(network_selections_mean < network_selections_mean.quantile(q=0.85))
            threshold_option = f'\n threshold > 85th percentile'
        elif type(threshold) == int: 
            network_selections_mean = network_selections_mean.mask(network_selections_mean < threshold)
            threshold_option = f'\n threshold > {threshold} interactions'
        else: 
            print('threshold type not valid')
    else:
        #network_selections_mean = network_selections_mean
        threshold_option = ''

    
    # 1 - plot selections matrix as heatmap 
    plt.clf()
    plt.figure(figsize=(8,8))
    sns.set(font_scale=2)
    s = sns.heatmap(network_selections_mean, cmap='gist_yarg', 
                    #viridis, seismic, coolwarm, cool, hot, PiYG, summer, PuOr, PRGn, Blues, Greens, Purples, PuBuGn, YlGnBu, BuGn, gist_yarg, RdPu, PuRd
                    xticklabels=network_selections_mean.index, yticklabels=network_selections_mean.columns, 
                    annot=False, linecolor='black', linewidths=0.003, square=True, 
                    vmin = 0, vmax = network_selections_mean.max(axis=None).max())
                    #vmin = 0, vmax = 60)

    s.set(ylabel='Player 1 (selectOR)', xlabel='Player 2 (selectED)',
          title=f'Selection dynamics across players \n in population {population_tile_forpaper} \n (# times selected) \n (over episodes {episode_range[0]+1}-{episode_range[1]}) {threshold_option}')
    #manually set limits to top and bottom of heatmap do not get cut off 
    bottom, top = s.get_ylim()
    s.set_ylim(bottom+0.5, top-0.5)

    plt.savefig(f'outcome_plots/matrix_selections_bipartite_byplayertype_popul{population_tile_forpaper}_threshold{threshold}.pdf', bbox_inches='tight')
    plt.show()

    # 2 - plot selections as network plot, circular layout 
    plt.clf()
    plt.figure(figsize=(7,6))
    temp = network_selections_mean.fillna(0).astype('float')     #temp = network_selections_mean.replace(0, np.nan).astype('float')
    if normalise: 
        #min-max normalise for the network plot to appear more neat 
        network_selections_mean_fornetworkplotting = temp.sub(temp.min().min()).div( (temp.max().max()-temp.min().min()) )
        network_selections_mean_fornetworkplotting = network_selections_mean_fornetworkplotting.mul(5)

        #max-normalise
        #network_selections_mean_fornetworkplotting = temp.div( temp.max().max() )
    else: 
        network_selections_mean_fornetworkplotting = temp

    G=nx.from_pandas_adjacency(network_selections_mean_fornetworkplotting.astype('float'), create_using=nx.DiGraph())
    #G = nx.DiGraph(network_selections_mean_fornetworkplotting.astype('float'))

    widths = nx.get_edge_attributes(G, 'weight')
    #node_colors = ['grey' for node in G.nodes()]
    player_types = [node.split('_')[1] for node in G.nodes()]
    node_colors = [color_mapping_longtitle[title_mapping_forpaper_shorttolong[player]] for player in player_types]
    #node_colors = [color_mapping_longtitle[title_mapping_forpaper_shorttolong[node]] for node in G.nodes()]
    
    edges = {k: v for k, v in widths.items() if not isnan(v)}
    pos = nx.spring_layout(G, k=1, seed=15) #seed 5  # positions for all nodes - seed for reproducibility
    #pos = nx.circular_layout(G) #, scale=5, center=('S', 'DE')  # positions for all nodes - seed for reproducibility
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color=node_colors)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=list(edges.keys()), width=list(edges.values()), alpha=0.5, 
                           edge_color="black", style="solid", arrows=True, arrowstyle='-|>', connectionstyle="arc3,rad=0.1", node_size=1200)
    # node labels
    #nodenames = {n:f'{n.split('_')[0]} \n {n.split('_')[1]} \n' for n in G.nodes()}
    nodenames = {n:n.split('_')[0]+'\n'+n.split('_')[1] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=nodenames, font_size=15, font_family="serif")

    plt.title(f'Selections \n in population {population_tile_forpaper} {threshold_option}')
    plt.savefig(f'outcome_plots/network_selections_bipartite_byplayertype_popul{population_tile_forpaper}_threshold{threshold}_normalise{normalise}{run_option}.pdf', bbox_inches='tight')
    plt.show()



def calculate_network_selection_bipartite_onerun(destination_folder, episode_idxs, num_episodes, run, reorder_index=False):

    titles = get_titles_for_population(destination_folder)
    population_size = len(titles)
    destination_folder = 'results/' + destination_folder
    long_titles = [title_mapping[title] for title in titles]
    print('for the chart label, assuming only one player is the majority player in whole population!!')
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]
    population_tile_forpaper = f'majority {majority_title}'
    print(f'running for populaiton {population_tile_forpaper}')

    #loop over eveyr run 
    #network_selections = pd.DataFrame(columns=player_types_forpaper, index=player_types_forpaper)
    network_selections_everyrun = dict.fromkeys(list(range(1)))

    # for a single run: 
    df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'idx_p1', 'title_p1', 'idx_p2', 'title_p2']]
    small_df = df.loc[ df['episode'].isin(episode_idxs) ]
    small_df.drop('episode', axis='columns', inplace=True)
    #replace long player title wiht short 
    small_df.replace({'title_p1': title_mapping_forpaper_long, 'title_p2': title_mapping_forpaper_long}, inplace=True)
    #reorder index so prosocial players appear first! 
    if reorder_index:
        relabel_dict = relabel_player_index(long_titles)
        small_df['idx_p1'] = small_df['idx_p1'].map(relabel_dict)
        small_df['idx_p2'] = small_df['idx_p2'].map(relabel_dict)

    #concatenate type and index for bipartite graph 
    small_df['title_idx_p1'] = small_df['idx_p1'].astype(str) + '_' + small_df['title_p1']
    small_df['title_idx_p2'] = small_df['idx_p2'].astype(str) + '_' + small_df['title_p2']

    #create unique ordered index for this population 
    labels_series = small_df[['idx_p1','title_p1']].value_counts()
    labels = ['{}_{}'.format(i, j) for i, j in labels_series.index]

    small_df["value"]=1
    network_selections_matrix_temp = pd.crosstab(small_df['title_idx_p1'], small_df['title_idx_p2'].fillna(0))
    #add columns for all labels, even those that were not present in the selections at all
    network_selections_matrix = pd.DataFrame(index=labels, columns=labels)
    #reorder columns and rows so they are in fixed order
    #network_selections_matrix = network_selections_matrix[labels]
    network_selections_matrix = network_selections_matrix.reindex(labels)
    #add crosstab values into results df with all possible columns 
    network_selections_matrix[network_selections_matrix_temp.columns] = network_selections_matrix_temp

    #store results for this run in big df 
    network_selections_everyrun[run] = network_selections_matrix
            
    #average numbers across all runs
    network_selections_mean = pd.concat(network_selections_everyrun.values()).reset_index().groupby("index").mean()
    #reorder index and columns in result 
    network_selections_mean_ordered = network_selections_mean.reindex(index=labels, columns=labels)

    if False: 
        #if reorder_index:
        print('TO FINISH reorder_index')
        my_list = [i.split('_') for i in network_selections_mean_ordered.index]
        my_df = pd.DataFrame(my_list, columns=['idx_p1', 'title_p1'])

        #get target index in the desired order 
        new_list = reorder_selections_index(my_df)
        my_df.set_index('title_p1', inplace=True)

        
        new_idx_df = pd.DataFrame(new_list, columns=['title_p1'])
        new_idx_df['new_idx'] = new_idx_df.index.astype(str)
        #new_idx_df['new_idx_title'] = new_idx_df.index.astype(str) + '_' + new_idx_df['title_p1']
        new_idx_df.set_index('title_p1', inplace=True)

        final_idx_df = my_df.join(new_idx_df)
        #final_idx_df['index'] = final_idx_df['idx_p1'] + '_' + final_idx_df.index.astype(str)
        #final_idx_df.set_index('index', inplace=True)
        #final_idx_df.drop('idx_p1', axis=1, inplace=True)
        final_idx_df.set_index('idx_p1', inplace=True)
        final_idx_df.to_dict()

        #my_df.reindex(new_list)
        network_selections_mean_ordered = network_selections_mean_ordered.join(final_idx_df)

        #reorder original df 
        network_selections_mean_ordered.reindex(index=new_list, columns=new_list) 

    #store results in this population's results folder 
    network_selections_mean_ordered.to_csv(f'{destination_folder}/network_selections_bipartite_mean_episodes{episode_range}_run{run}.csv')

def relabel_player_index(long_titles):
    #my_list = [i.split('_') for i in network_selections_mean_ordered.index]
    #my_df = pd.DataFrame(my_list, columns=['idx_p1', 'title_p1'])

    #get target index in the desired order
    sort_order = ['S', 'Ut', 'De', 'V-Eq', 'V-Ki', 'aUt', 'mDe', 'V-In', 'V-Ag']
    original_titles = [title_mapping_forpaper_long[title] for title in long_titles]
    original_df = pd.DataFrame(original_titles, columns=['title_p1'])
    #original_df['title_p1_enum'] = lambda x: x.enumerate() original_df['title_p1']
    duplicated_titles = original_df[original_df['title_p1'].duplicated()]
    duplicate_single = duplicated_titles.iloc[0,0]
    #handle duplicate title in original_df
    original_df['title_p1_identifier'] = ''    
    original_df.loc[original_df['title_p1']==duplicate_single, 'title_p1_identifier'] = np.arange((original_df['title_p1']==duplicate_single).sum()) + 1
    original_df['title_p1_formatted'] = original_df['title_p1'] + original_df['title_p1_identifier'].astype(str)
    original_df.drop(['title_p1', 'title_p1_identifier'], axis=1, inplace=True)

    original_titles.sort(key = lambda i: sort_order.index(i)) # note this happens inplace 
    new_idx_df = pd.DataFrame(original_titles, columns=['title_p1'])
    new_idx_df['new_idx'] = new_idx_df.index.astype(str)
    #handle duplicate title in new_idx_df
    new_idx_df['title_p1_identifier'] = ''    
    new_idx_df.loc[new_idx_df['title_p1']==duplicate_single, 'title_p1_identifier'] = np.arange((new_idx_df['title_p1']==duplicate_single).sum()) + 1
    new_idx_df['title_p1_formatted'] = new_idx_df['title_p1'] + new_idx_df['title_p1_identifier'].astype(str)
    new_idx_df.drop(['title_p1', 'title_p1_identifier'], axis=1, inplace=True)

    new_idx_df.set_index('title_p1_formatted', inplace=True)

    #new_list = reorder_selections_index(set(small_df['title_p1']))
    final_df = original_df.reset_index(inplace=False).set_index('title_p1_formatted', inplace=False)
    final_df = final_df.join(new_idx_df) #this currently stores the original index 
    #final_df.set_index('index', inplace=True)
    final_dict = dict(zip(final_df['index'], final_df['new_idx'].astype(int))) 

    return final_dict

        

#def reorder_selections_index(labels_df):
#    my_list = list(pd.DataFrame(labels_df).reset_index()['title_p1'])
#    sort_order = ['S', 'Ut', 'De', 'V-Eq', 'V-Ki', 'aUt', 'mDe', 'V-In', 'V-Ag']

#    my_list.sort(key = lambda i: sort_order.index(i))

#    return my_list


############################################################################
#### after IMPLEMENTING EPISODES & fixing sel memory refresh ####
############################################################################

game_title = 'IPD'


#os.chdir('EPISODES_PartnerSelection (statenotpairs, fullobservability, fixedSelmemory) ')
#os.chdir('EPISODES_PartnerSelection (statenotpairs, fullobservability) ')

record_Loss = 'both' 
record_Qvalues = 'both'




num_iter=30000 #this now represents the number of episodes (rather than original iterations) 




destination_folder = '3xS_TEST'
n_runs = 1 
num_iter = 1000


if Testing_Baseline: 

    destination_folder = '3xS___iter30000_runs15'
    n_runs = 15


    destination_folder = '5xS___iter30000_runs15'
    n_runs = 15


    destination_folder = '20xS___iter30000_runs15'
    n_runs = 15

    #destination_folder = '3xS___iter30000_runs5'
    #n_runs = 5 



    #### check LR faster ###
    os.chdir('../_LR0.005')
    num_iter = 15000

    destination_folder = '3xS___iter15000_runs10'
    n_runs = 10

    destination_folder = '5xS___iter15000_runs10'
    n_runs = 10

    destination_folder = '20xS___iter15000_runs10'
    n_runs = 10

    num_iter = 30000
    destination_folder = '20xS___iter30000_runs10'
    n_runs = 5


    os.chdir('../_LR0.01')
    num_iter = 15000

    destination_folder = '20xS___iter15000_runs10'
    n_runs = 10 


    os.chdir('../_LR0.003')
    num_iter = 15000

    destination_folder = '5xS___iter15000_runs5'
    n_runs = 5

    destination_folder = '1xS_10xAC_10xAD___iter15000_runs5'
    n_runs = 2

    destination_folder = '20xS___iter15000_runs5'
    n_runs = 5 



    os.chdir('../_LR0.1')
    num_iter = 15000
    n_runs = 5

    destination_folder = '5xS___iter15000_runs5'


    destination_folder = '1xS_10xAC_10xAD___iter15000_runs5'
    n_runs = 2

    destination_folder = '20xS___iter15000_runs5'



    #### check static opponents ####
    os.chdir('../EPISODES_PartnerSelection (statenotpairs, fullobservability, fixedSelmemory) ')
    n_runs = 10
    num_iter = 10000

    destination_folder = '1xS_3xAD_1xAC___iter10000_runs10'
    destination_folder = '1xS_2xAC_2xAD___iter10000_runs10'

    #destination_folder = '1xS_1xCD_1xAC_1xAD___iter10000_runs10'
    #destination_folder = '1xS_1xGEa_1xGEb_1xGEc___iter10000_runs10'


    num_iter = 30000

    destination_folder = '1xS_1xCD_1xAC_1xAD___iter30000_runs10'
    destination_folder = '1xS_1xGEa_1xGEb_1xGEc___iter30000_runs10'

    destination_folder = '1xS_5xAC_5xAD___iter30000_runs10'
    destination_folder = '1xS_10xAC_10xAD___iter30000_runs10'
    n_runs = 4


    destination_folder = '1xS_1xGEd_1xGEe___iter30000_runs10'
    n_runs = 10
    num_iter = 30000

    destination_folder = '1xS_1xGEa_1xGEb_1xGEc_1xGEd_1xGEe___iter30000_runs10'



    #### check other parameters ####
    os.chdir('../_LRdil0.001_LRsel0.005')

    os.chdir('../_payoffmatwith0')

    os.chdir('../_player2learns_False')


    destination_folder = '5xS___iter15000_runs5'
    n_runs = 5

    destination_folder = '1xS_10xAC_10xAD___iter15000_runs5'
    n_runs = 2

    destination_folder = '20xS___iter15000_runs5'
    n_runs = 5

    #TO DO 
    destination_folder = '20xS___iter30000_runs5'
    n_runs = 1
    num_iter = 30000


    if False: #from before fixing sel memory refresh 
        #### fixed Qvalue storage 
        os.chdir('_fixedQvalstorage')
        destination_folder = '3xS___iter40000_runs10'
        n_runs = 10
        num_iter=40000


        #### LR larger 
        os.chdir('_fixedQvalstorage, LR0.01')

        destination_folder = '3xS___iter20000_runs5'
        n_runs = 5
        destination_folder = '5xS___iter20000_runs5'

    if False: #from before 
        ############################################################################
        #### since changing state to notpairs
        ############################################################################

        game_title = 'IPD'
        record_Loss = 'both'
        record_Qvalues = 'both'

        #### BASELINE - b:c=2:1 ####
        os.chdir('../_baseline')

        n_runs=20
        num_iter=100000
        #population_size=3

        destination_folder = '3xS__'
        destination_folder = '5xS__'
        destination_folder = '20xS__'

        num_iter=500000
        n_runs = 10
        destination_folder = '5xS___iter500000_runs10'
        destination_folder = '20xS___iter500000_runs10'


        #### b:c=3:1 ####
        n_runs=10
        num_iter = 100000
        os.chdir('_bc3')

        destination_folder = '5xS___runs10'
        destination_folder = '20xS___runs10'
        destination_folder = ''


        num_iter=500000
        n_runs=10
        destination_folder = '5xS___iter500000_runs10'
        destination_folder = '20xS___iter500000_runs10'


        #### b:c=10:1 ####
        n_runs=10
        os.chdir('_bc10')

        destination_folder = '5xS___runs10'
        destination_folder = '20xS___runs10'
        destination_folder = ''

        num_iter=500000
        destination_folder = '5xS___iter500000_runs10'
        destination_folder = '20xS___iter500000_runs10'


        #### b:c=20:1 ####
        n_runs=10
        os.chdir('_bc20')

        destination_folder = '5xS___runs10'
        destination_folder = '20xS___runs10'
        destination_folder = ''

        num_iter = 500000
        destination_folder = '5xS___iter500000_runs10'
        n_runs = 8 
        destination_folder = '20xS___iter500000_runs10'
        n_runs = 7


        #### check Selfish vs static opponents on very small population to strip away the complexity #### 
        os.chdir('_staticsmallchecks')
        #os.chdir('NEWEST_IPD_DQN_PartnerSelection (statenotpairs, fullobservability) ')
        destination_folder  ='1xS_1xAC__'
        destination_folder  ='1xS_1xAD__'
        destination_folder = '2xS__'
        destination_folder = '2xS___runs5'
        population_size=2
        titles = ['QLS', 'AC']
        opponents = ['AC']


        destination_folder  ='1xS_1xAC_1xAD__'
        destination_folder = '1xS_1xAC_1xAD___runs5'
        titles = ['QLS', 'AC', 'AD']
        oppponents = ['AC', 'AD']

        destination_folder = '3xS__'
        destination_folder = '3xS___runs5'
        titles = ['QLS', 'QLS', 'QLS']
        opponents = ['Selfish1', 'Selfish2']
        population_size=3

        num_iter = 100000
        n_runs = 20




        destination_folder = '3xS___iter300000_runs10'
        num_iter = 300000
        n_runs = 10
        opponents = ['Selfish1', 'Selfish2']


        destination_folder = '10xS___runs10'
        num_iter = 100000
        n_runs = 10
        opponents = ['Selfish1', 'Selfish2', 'Selfish3', 'Selfish4', 'Selfish5', 'Selfish6', 'Selfish7', 'Selfish8', 'Selfish9', 'Selfish10']



        destination_folder = 'results/' + destination_folder

        for run_idx in range(1):
            df = pd.read_csv(f'{destination_folder}/history/run{run_idx}.csv', index_col=0)
            #print(df[df[f'title_p{1}']=='Selfish'][f'action_player{1}'].value_counts())
            #print(df[df[f'title_p{2}']=='Selfish'][f'action_player{2}'].value_counts())
            
            for p_idx in range(population_size):
                print('player {p_idx}, as player1 or player2')
                print(df[df[f'idx_p{1}']==1][f'action_player{1}'].value_counts())
                print(df[df[f'idx_p{2}']==2][f'action_player{2}'].value_counts())
                print(' ')


        if False: 
            player_idx = 0 
            run_idx = 1
            which = 'local'
            plot_Q_values_population_statenotpairs(destination_folder, population_size, 1, which, record_Qvalues, maxq=5, fewer_than_nruns = False)


    #### try even smaller & mixed LR ####
    os.chdir('_LR0.0001')
    n_runs = 5 
    num_iter = 15000 

    destination_folder = '1xS_10xAC_10xAD___iter15000_runs5'

    destination_folder = '20xS___iter15000_runs5'


    os.chdir('../_LRsel0.005_LRdil0.001')
    n_runs = 5
    num_iter = 15000

    destination_folder = '1xS_10xAC_10xAD___iter15000_runs5'

    destination_folder = '20xS___iter15000_runs5'


    os.chdir('../_LRsel0.001_LRdil0.005')


    os.chdir('../_LRsel0.001_LRdil0.01')


    os.chdir('../_LRsel0.01_LRdil0.001')
    destination_folder = '20xS___iter30000_runs5'
    n_runs = 5
    num_iter = 30000 


    os.chdir('../_LRsel0.001_LRdil0.0001')
    n_runs = 5
    num_iter = 30000
    destination_folder = '20xS___iter30000_runs5'

    destination_folder = '1xS_10xAC_10xAD___iter30000_runs5'


    os.chdir('../_LRsel0.002_LRdil0.0001')


    os.chdir('_LRsel0.02_LRdil0.001')
    destination_folder = '20xS___iter30000_runs5'
    n_runs = 5
    num_iter = 30000 







    #### try random matching ####
    os.chdir('../_randommatch')
    n_runs = 5 
    num_iter = 15000 

    destination_folder = '1xS_10xAC_10xAD___iter15000_runs5'

    destination_folder = '5xS___iter15000_runs5'

    destination_folder = '20xS___iter15000_runs5'


    os.chdir('_randommatch_LRdil0.01')
    n_runs = 1
    num_iter = 30000 
    destination_folder = '20xS___iter30000_runs5'



    #### choosing the sel:dil rate ratio --> how many dilemmas does a player play on average per spisode, within the first 15K episodes? ####
    os.chdir('../EPISODES_PartnerSelection (statenotpairs, fullobservability, fixedSelmemory) ')
    destination_folder = '20xS___iter30000_runs15'
    n_runs = 15 
    num_iter = 30000

    os.chdir('../_payoffmatwith0')
    destination_folder = '20xS___iter30000_runs5'
    n_runs = 1

    counts_per_run = calculate_dil_per_episode(destination_folder, n_runs, population_size, cutoff=[0,15000]])
    counts_per_run.mean()

    run1 = pd.read_csv(f'{destination_folder}/history/run1.csv', index_col=0)
    counts_selected = pd.DataFrame(run1[0:15000*population_size]['idx_p2'].value_counts())
    counts_selected.columns = ['n_times_selected']
    per_episode = counts_selected / 15000 
    per_episode.hist()
    plt.title('No times each player gets selected per episode \n within the firt 15K episodes \n (run1)')

    run1_small = run1[['episode', 'idx_p2']][0:15000*population_size]
    counts_per_episode = pd.DataFrame(run1_small.groupby(['episode', 'idx_p2']).size())
    counts_per_episode.columns = ['count_per_episode']

    #counts_per_episode.reset_index(inplace=True)
    counts_per_episode.groupby('idx_p2').mean('count_per_episode').hist()
    plt.title('Average number of times each player gets selected per episode \n within the firt 15K episodes \n (run1)')

    counts_per_episode.groupby('idx_p2').mean('count_per_episode').mean()


    #LRsel != LRsel, larger values 
    #TO DO 
    os.chdir('_LRsel0.05_LRdil0.005')
    os.chdir('../_LRsel0.1_LRdil0.01')
    os.chdir('../_LRsel0.05_LRdil0.0025')
    os.chdir('../_LRsel0.1_LRdil0.005')
    os.chdir('../_LRsel0.5_LRdil0.025') 

    os.chdir('../_payoffmatwith0_LRsel0.005_LRdil0.001')
    destination_folder = '20xS___iter10000_runs1' 
    n_runs = 1 
    num_iter = 10000

    os.chdir('_payoffmatwith0_LRsel0.02_LRdil0.001')
    destination_folder = '20xS___iter30000_runs5' 
    n_runs = 5
    num_iter = 30000


    os.chdir('../_payoffmatwith0_LRsel0.05_LRdil0.005')
    os.chdir('../_payoffmatwith0_LRsel0.1_LRdil0.01')
    os.chdir('../_payoffmatwith0_LRsel0.05_LRdil0.0025')  
    os.chdir('../_payoffmatwith0_LRsel0.1_LRdil0.005')


    os.chdir('../_payoffmatwith0_LRsel0.05_LRdil0.025')
    os.chdir('../_payoffmatwith0_LRsel0.05_LRdil0.01')
    os.chdir('../_payoffmatwith0_LRsel0.2_LRdil0.01')




    #analyse  results with freezing network - 3 update_every param,eters; 2 LR combinations 
    os.chdir('_freezenet5_payoffmatwith0_LRsel0.005_LRdil0.001')
    os.chdir('../_freezenet10_payoffmatwith0_LRsel0.005_LRdil0.001')
    os.chdir('../_freezenet20_payoffmatwith0_LRsel0.005_LRdil0.001')

    os.chdir('_freezenet5_payoffmatwith0_LRsel0.02_LRdil0.001')
    os.chdir('../_freezenet10_payoffmatwith0_LRsel0.02_LRdil0.001')
    os.chdir('../_freezenet20_payoffmatwith0_LRsel0.02_LRdil0.001')

    #analyse 2xS only 
    os.chdir('_freezenet5_payoffmatwith0_LRsel0.005_LRdil0.001')
    os.chdir('../_2xS_randommatch_freezenet5_payoffmatwith0_LRsel0.005_LRdil0.001')
    os.chdir('../_2xS_bc3_freezenet5_LRsel0.005_LRdil0.001')
    os.chdir('../_2xS_bc2_freezenet5_LRsel0.005_LRdil0.001')

    destination_folder = '2xS___iter30000_runs5'

    #analyse 5xS and 10xS for the selected parameters (freeze5) 
    os.chdir('_freezenet5_payoffmatwith0_LRsel0.005_LRdil0.001')
    destination_folder = '5xS___iter30000_runs5'
    destination_folder = '10xS___iter30000_runs5'

    #analyse 20xSrun for longer for parameters _payoffmatwith0_LRsel0.02_LRdil0.00
    os.chdir('../_payoffmatwith0_LRsel0.02_LRdil0.001')
    destination_folder = '20xS___iter50000_runs5'
    num_iter = 50000 


    #analyse resutls where we freeze only the dilemma network 
    os.chdir('_dilfreezenet5_payoffmatwith0_LRsel0.005_LRdil0.001')
    os.chdir('../_dilfreezenet10_payoffmatwith0_LRsel0.005_LRdil0.001')

    os.chdir('../_dilfreezenet5_payoffmatwith0_LRsel0.02_LRdil0.001')
    os.chdir('../_dilfreezenet10_payoffmatwith0_LRsel0.02_LRdil0.001')


    #analyse resutls with identity
    os.chdir('../_withidentity_freeze1_with0_LRsel0.005_LRdil0.001')
    destination_folder = '5xS___iter30000_runs5'
    destination_folder = '10xS___iter30000_runs5'
    destination_folder = '20xS___iter30000_runs5'

    os.chdir('_withidentity_freeze5_with0_LRsel0.005_LRdil0.001')
    destination_folder = '5xS___iter30000_runs5'
    destination_folder = '10xS___iter30000_runs5'
    destination_folder = '20xS___iter30000_runs5'



    #analyze without a target network
    os.chdir('../_notargetnet')

    #analyse results with  LR0.0001 - with target network
    os.chdir('../_LR0.0001_new')

    #analyse resutls without identity, with epsdecay 
    os.chdir('../_epsdecay_freeze1_with0_LRsel0.005_LRdil0.001') 
    destination_folder = '5xS___iter30000_runs5'
    destination_folder = '10xS___iter30000_runs5'
    destination_folder = '20xS___iter30000_runs5'


    #analye results with b:c=20:1 
    os.chdir('_bc20')

    #analyse results with LR0.0005 
    os.chdir('../_LR0.0005_new') 

    #analyse without a target network and with slower LR (this should be like Nicolas's version, 
    # except we are not averaging loss across all runs in an apisode)
    os.chdir('../_notargetnet_LR0.0001')
    
    #analyse results with average loss across all dilemma games in an episode + after fixing other things & warning with torch shapes 
    os.chdir('../NEW_payoffmatwith0_LRsel0.005_LRdil0.001')

    os.chdir('../NEW_LR0.0005')

    os.chdir('_notargetnet')


#PLOTTING DIRECTLY ON REMOTE CLUSTER 
#connec tto SSH within VSCode: Cmd+Shift+P -> tails.cs.ucl.ac.uk -> enter password 
#connect to SSH within the terminal here: 'ssh -t lkarmann@tails.cs.ucl.ac.uk ssh -t geogcpu3.cs.ucl.ac.uk'
#connect to SSH within VSCode - 'ssh -t lkarmann@tails.cs.ucl.ac.uk ssh -t geogcpu3.cs.ucl.ac.uk'

os.getcwd()
os.listdir()
#os.chdir('/cs/research/ntelsys/big_experiment_data/my_env')

if other: #earlier analyses before I ran the main bulk of experimetns 
    #Baseline 
    destination_folder = '20xS___iter30000_runs20'

    #Prosocial only
    destination_folder = '1xUT_1xDE_1xVEe_1xVEk___iter30000_runs20'
    destination_folder = '2xUT_2xDE_2xVEe_2xVEk___iter30000_runs20'
    destination_folder = '4xUT_4xDE_4xVEe_4xVEk___iter30000_runs20'

    #Prosocial and Selfish only 
    destination_folder = '1xS_1xUT_1xDE_1xVEe_1xVEk___iter30000_runs20'

    destination_folder = '4xS_4xUT_4xDE_4xVEe_4xVEk_4xAL___iter30000_runs20'

    destination_folder = '10xS_2xUT_2xDE_2xVEe_2xVEk_2xAL___iter30000_runs20'

    destination_folder = '1xUT_1xDE_1xVEe_1xVEk_1xAL___iter30000_runs20' 

    destination_folder = '1xS_1xUT_1xDE_1xVEe_1xVEk_1xAL___iter30000_runs20'

    #Antisocial only
    destination_folder = '1xaUT_1xmDE_1xVie_1xVagg_1xaAL___iter30000_runs20'

    destination_folder = '2xaUT_2xmDE_2xVie_2xVagg_2xaAL___iter30000_runs20' 

    destination_folder = '4xaUT_4xmDE_4xVie_4xVagg_4xaAL___iter30000_runs20' 

    #Antisocial and Selfish only
    destination_folder = '4xS_4xaUT_4xmDE_4xVie_4xVagg_4xaAL___iter30000_runs20' 

    destination_folder = '10xS_2xaUT_2xmDE_2xVie_2xVagg_2xaAL___iter30000_runs20' 


    #Mixed - uniform 
    destination_folder = '1xUT_1xDE_1xVEe_1xVEk_1xAL_1xaUT_1xmDE_1xVie_1xVagg_1xaAL___iter30000_runs20' 
    destination_folder = '2xUT_2xDE_2xVEe_2xVEk_2xAL_2xaUT_2xmDE_2xVie_2xVagg_2xaAL___iter30000_runs20'

    destination_folder = '1xS_1xUT_1xDE_1xVEe_1xVEk_1xAL_1xaUT_1xmDE_1xVie_1xVagg_1xaAL___iter30000_runs20'
    destination_folder = '2xS_2xUT_2xDE_2xVEe_2xVEk_2xAL_2xaUT_2xmDE_2xVie_2xVagg_2xaAL___iter30000_runs20' 

    #Mixed, subpop - uniform 
    destination_folder = '1xS_1xUT_1xDE_1xaUT_1xmDE___iter30000_runs20'
    destination_folder = '3xS_3xUT_3xDE_3xaUT_3xmDE___iter30000_runs20'

    #Mixed, subpop - non-uniform 
    destination_folder = '8xS_3xUT_3xDE_3xaUT_3xmDE___iter30000_runs20'
    destination_folder = '3xS_8xUT_3xDE_3xaUT_3xmDE___iter30000_runs20' 
    destination_folder = '3xS_3xUT_8xDE_3xaUT_3xmDE___iter30000_runs20' 
    destination_folder = '3xS_3xUT_3xDE_8xaUT_3xmDE___iter30000_runs20'
    destination_folder = '3xS_3xUT_3xDE_3xaUT_8xmDE___iter30000_runs20' 

    destination_folder = '6xS_2xUT_2xDE_2xaUT_2xmDE___iter30000_runs20' 
    destination_folder = '2xS_6xUT_2xDE_2xaUT_2xmDE___iter30000_runs20' 
    destination_folder = '2xS_2xUT_6xDE_2xaUT_2xmDE___iter30000_runs20'
    destination_folder = '2xS_2xUT_2xDE_6xaUT_2xmDE___iter30000_runs20' 
    destination_folder = '2xS_2xUT_2xDE_2xaUT_6xmDE___iter30000_runs20' 


    destination_folder = '2xS_6xUT_6xDE_3xaUT_3xmDE___iter30000_runs20'
    destination_folder = '2xS_3xUT_3xDE_6xaUT_6xmDE___iter30000_runs20'



    #For counting popularity of different players 
    if counting_popularity:
            
        counts_per_run = calculate_dil_per_episode(destination_folder, n_runs, population_size, cutoff=[0,30000])
        counts_per_run.mean()

        run1 = pd.read_csv(f'{destination_folder}/history/run1.csv', index_col=0)
        counts_selected = pd.DataFrame(run1[0:130000*population_size]['idx_p2'].value_counts())
        counts_selected.columns = ['n_times_selected']
        per_episode = counts_selected / 30000 
        per_episode.hist()
        plt.title('No times each player gets selected per episode \n within the firt 15K episodes \n (run1)')

        run1_small = run1[['episode', 'idx_p2']][0:30000*population_size]
        counts_per_episode = pd.DataFrame(run1_small.groupby(['episode', 'idx_p2']).size())
        counts_per_episode.columns = ['count_per_episode']

        #counts_per_episode.reset_index(inplace=True)
        counts_per_episode.groupby('idx_p2').mean('count_per_episode').hist()
        plt.title('Average number of times each player gets selected per episode \n within the firt 15K episodes \n (run1)')

        counts_per_episode.groupby('idx_p2').mean('count_per_episode').mean()

    #check the impact of xi parameter for DE players 
    os.chdir('xi3')
    destination_folder = '6xS_2xUT_2xDE_2xaUT_2xmDE___iter30000_runs10'

    os.chdir('../xi1')
        
    #check the impact of randdommatch 
    os.chdir('randommatch')
    destination_folder = '6xS_2xUT_2xDE_2xaUT_2xmDE___iter30000_runs20'


    #analyse sub-population1 of size 16, non-uniform
    os.chdir('../EPISODES_PartnerSelection')

    destination_folder = '8xS_2xUT_2xDE_2xaUT_2xmDE___iter30000_runs20'
    destination_folder = '2xS_8xUT_2xDE_2xaUT_2xmDE___iter30000_runs20'
    destination_folder = '2xS_2xUT_8xDE_2xaUT_2xmDE___iter30000_runs20'
    destination_folder = '2xS_2xUT_2xDE_8xaUT_2xmDE___iter30000_runs20'
    destination_folder = '2xS_2xUT_2xDE_2xaUT_8xmDE___iter30000_runs20'


    #analyse sub-population2 of size 16, non-uniform
    destination_folder = '8xS_2xVEe_2xVEk_2xVie_2xVagg___iter30000_runs20'
    destination_folder = '2xS_8xVEe_2xVEk_2xVie_2xVagg___iter30000_runs20'
    destination_folder = '2xS_2xVEe_8xVEk_2xVie_2xVagg___iter30000_runs20'
    destination_folder = '2xS_2xVEe_2xVEk_8xVie_2xVagg___iter30000_runs20'
    destination_folder = '2xS_2xVEe_2xVEk_2xVie_8xVagg___iter30000_runs20'


#### PART 1 = plotting for the full mixed populations: conseq+norm+selfish
#if plotting from local directory 
os.getcwd()
os.chdir('EPISODES_PartnerSelection/conseq+norm+selfish')

#if plotting form external hard drive
os.getcwd()
os.chdir("/Volumes/G-DRIVE mobile USB-C/Partner Selection/EPISODES_IPD_PartnerSelection/conseq+norm_selfish")
os.listdir()


destination_folder = '8xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 

destination_folder = '1xS_8xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 
destination_folder = '1xS_1xUT_8xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 

destination_folder = '1xS_1xUT_1xaUT_8xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 
destination_folder = '1xS_1xUT_1xaUT_1xDE_8xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 

destination_folder = '1xS_1xUT_1xaUT_1xDE_1xmDE_8xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 
destination_folder = '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_8xVie_1xVEk_1xVagg___iter30000_runs20' 

destination_folder = '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_8xVEk_1xVagg___iter30000_runs20' 
destination_folder = '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_8xVagg___iter30000_runs20' 


#### PART 1.1 - full mixed populaiton but uniform 


destination_folder = '2xUT_2xaUT_2xDE_2xmDE_2xVEe_2xVie_2xVEk_2xVagg___iter30000_runs20' 
destination_folder = '2xS_2xUT_2xaUT_2xDE_2xmDE_2xVEe_2xVie_2xVEk_2xVagg___iter30000_runs20' 
#destination_folder = '1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 



#################################################################################
#### preparation to run for every destination_folder ####

num_iter = 30000

n_runs = len(glob.glob('results/'+destination_folder+'/history/*.csv'))
print('n_runs = ', n_runs)

titles = get_titles_for_population(destination_folder)

population_size = len(titles)
destination_folder = 'results/' + destination_folder

long_titles = [title_mapping[title] for title in titles]


#################################################################################

reformat_a_for_population(destination_folder, n_runs, population_size, num_iter, long_titles) 

reformat_reward_for_population(destination_folder, n_runs, population_size, long_titles)

#################################################################################
#### functions to run for every destination_folder ####


#### EPISODE version ####
episode_plot_cooperative_selections_whereavailable(destination_folder, titles, n_runs)
#episode_plot_cooperative_selections_perrun(destination_folder, titles, n_runs) #each run separately 

#
episode_plot_cooperation_population_v2(destination_folder, titles, n_runs, num_iter, with_CI=True, reduced=True)
episode_plot_cooperation_perrun(destination_folder, titles, n_runs, num_iter) #each run separately 

#
episode_plot_action_pairs_population_new(destination_folder, titles, n_runs, option=None, combine_CDDC=False)


episode_plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, num_iter, population_size) 

#
plot_actions_aggbyplayertype_avgacrossruns(destination_folder, titles, n_runs, num_iter, population_size)

episode_plot_reward_eachplayerinpopulation()

#
episode_plot_collective_outcomes_population(destination_folder, titles, n_runs, num_iter, population_size)



#analyse how often each opponent is selected
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range='full') 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[500]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[2500]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[5000]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[7500]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[10000]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[20000]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[30000]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[29900,30000]) 




#################################################################################
#### plot ACROSS MANY POPULATIONS  ####
sns.set_style("darkgrid")
num_iter = 30000


POPULATIONS = ['8xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20', 
               '1xS_8xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_8xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_8xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_8xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_1xmDE_8xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_8xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_8xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_8xVagg___iter30000_runs20']



#actions - % cooperation over time
manypopulations_calculate_cooperation_allplayers_avgacrossruns(POPULATIONS, num_iter)
manypopulations_plot_cooperation_allplayers_avgacrossruns(POPULATIONS, num_iter)

#actions only by the Selfish player over time 
movingN=1000
manypopulations_plot_cooperation_aggbyplayertype_avgacrossruns(POPULATIONS, num_iter, title='Selfish')

#actions by each player type 
for player_type in set(long_titles):
    manypopulations_plot_cooperation_aggbyplayertype_avgacrossruns(POPULATIONS, num_iter, title=player_type)

manypopulations_plot_cooperation_aggbyplayertype_avgacrossruns(POPULATIONS, num_iter, title='VirtueEthics_aggressiveness')
manypopulations_plot_cooperation_aggbyplayertype_avgacrossruns(POPULATIONS, num_iter, title='VirtueEthics_inequality')


#social outcomes over time
manypopulations_calculate_outcomes_allplayers_avgacrossruns(POPULATIONS, num_iter)
manypopulations_plot_outcomes_allplayers_avgacrossruns(POPULATIONS, num_iter)

#reward over time
print(set(long_titles))
title = 'VirtueEthics_aggressiveness' 
manypopulations_plot_reward_aggbyplayertype_avgacrossruns(POPULATIONS, titles, n_runs, num_iter, population_size, which='game', cumulative = False, title=title)
manypopulations_plot_reward_aggbyplayertype_avgacrossruns(POPULATIONS, titles, n_runs, num_iter, population_size, which='game', cumulative = True, title=title)
manypopulations_plot_reward_aggbyplayertype_avgacrossruns(POPULATIONS, titles, n_runs, num_iter, population_size, which='intr', cumulative = False, title=title)
manypopulations_plot_reward_aggbyplayertype_avgacrossruns(POPULATIONS, titles, n_runs, num_iter, population_size, which='intr', cumulative = True, title=title)
#agg = 'sum' #sum rewards per episode

#cumulative reward on the final iteration
manypopulations_plot_heatmap_reward(POPULATIONS, num_iter, reward_type='game')
manypopulations_plot_heatmap_reward(POPULATIONS, num_iter, reward_type='intr')

#selections towards the end of trainig - which player is the most poppular? 
manypopulations_calculate_heatmap_selections(POPULATIONS, num_iter, episode_range=[29900, 30000])
manypopulations_plot_heatmap_selections(POPULATIONS, num_iter, episode_range=[29900, 30000])

#selection network among player types
manypopulations_calculate_network_selections(POPULATIONS, num_iter, episode_range=[0, 30000]) 
for destination_folder in POPULATIONS:
    plot_network_selections(destination_folder, num_iter, episode_range=[0, 30000])

#selection network among specific players -bipartite graph 
manypopulations_calculate_network_selection_bipartite(POPULATIONS, num_iter, episode_range=[0, 30000])
for destination_folder in POPULATIONS:
    plot_network_selections_bipartite(destination_folder, num_iter, episode_range=[0, 30000], threshold=False, normalise=True)
#plot the above with threshold=True for network plots, threshold=False for the heatmap plots

########################################################################################
#### deep-dive into V_Ki player type - why the drop in cooperation? 
#selections towards the end of trainig - which player is the most poppular? 
destination_folder = POPULATIONS[-2] #majorty-V-Ki

episode_range = [0, 100]
episode_range = [5000, 5100]
episode_range = [7500, 7600]
episode_range = [9000, 9100]
episode_range = [12000, 12100]
episode_range = [29900, 30000]

episode_idxs = list(range(episode_range[0],episode_range[1]))
num_episodes = episode_range[1] - episode_range[0]

#plot across many runns, averafged 
calculate_network_selection_bipartite(destination_folder, episode_idxs, num_episodes)
plot_network_selections_bipartite(destination_folder, num_iter, episode_range, threshold=True, normalise=True)

#plot only for one run 
calculate_network_selection_bipartite_onerun(destination_folder, episode_idxs, num_episodes, run=1, reorder_index=True) 
plot_network_selections_bipartite(destination_folder, num_iter, episode_range, threshold=False, normalise=True, one_run='1')

#use population-specific functions: 
### NOTE run the preprocessing of destination_folder and titles first! 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range) 

palette = [color_mapping_longtitle[t] for t in long_titles]

orders = [] 
for idx in range(population_size):
    print(idx)
    order_idx = get_opponents_from_title(long_titles, population_size, title=long_titles[idx], idx=idx)
    orders.append(order_idx)

palettes = []
for idx in range(population_size):
    palette_temp = palette.copy()
    del palette_temp[idx]
    palettes.append(palette_temp)
    

idx = 13
title='VirtueEthics_kindness'
run = 1 

for idx in [9,10,11,12,13,14]:
    plot_opponents_selected(destination_folder, n_runs=1, palette=palettes[idx], order=orders[idx], long_titles=long_titles, title=title, idx=idx, episode_range=episode_range)

for idx in [7,8,9,10,11,12,13,14]:
    plot_actions_player_episoderange(destination_folder, run_idx=1, player_idx=idx, actions=['C','D'], title='VirtueEthics_kindness', color=color_mapping_longtitle[title])


#plot_actions_player_last100(destination_folder, n_runs=1, actions=['C', 'D'], title='VirtueEthics_kindness', idx=7)

#plot_all_actions_player(destination_folder, run_idx=1, player_idx=7, color=color_mapping_longtitle[title], title='VirtueEthics_kindness') # actions=['C','D']

#plot_actions_player(destination_folder, n_runs=1, actions=['C', 'D'], title='VirtueEthics_kindness', idx=7)

plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)



#analyse actions on the episodes of interest


if False: 

    destination_folder = POPULATIONS[-2] #majorty-V-Ki
    player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    print('NOTE assuming all player types are possible in population!!')
    player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]
    figsize=(6, 6)
    populations = ['majority-V-Ki']
    os.mkdir('V-Ki_deepdive')

    episode_range = [5000, 5001]


    episode_idxs = list(range(episode_range[0],episode_range[1]))
    num_episodes = episode_range[1] - episode_range[0] 
    majority_title, results_allruns_percenttimes_selected, ci_allruns_percenttimes_selected = calculate_heatmap_selections(destination_folder, num_iter, num_episodes, episode_idxs)

        
    matrix_selections = pd.DataFrame(columns=['majority-V-Ki'], index=player_types_forpaper)
    matrix_selections[f'majority-{majority_title}'] = results_allruns_percenttimes_selected.astype(float)   
    matrix_selections.to_csv(f'V-Ki_deepdive/matrix_selections_episodes{episode_range}.csv')

    #visualise results
    #plot_heatmap_selections(POPULATIONS, num_iter, episode_range=[0, 1])
    matrix_selections = pd.read_csv(f'V-Ki_deepdive/matrix_selections_episodes{episode_range}.csv', index_col=0)
    matrix_selections_percent = matrix_selections*100 

    #plot selections matrix
    plt.figure(figsize=figsize)
    sns.set(font_scale=2)
    s = sns.heatmap(matrix_selections_percent, cmap='gist_yarg', 
                        #viridis, seismic, coolwarm, cool, hot, PiYG, summer, PuOr, PRGn, Blues, Greens, Purples, PuBuGn, YlGnBu, BuGn, gist_yarg, RdPu, PuRd
                        xticklabels=populations, yticklabels=player_types_forpaper, 
                        annot=False, linecolor='black', linewidths=0, square=True, vmin = 0, vmax = 100)
    s.set(xlabel='Population', ylabel='Player Type',
            title=f'Popularity of each player type \n (% times selected) \n (over last 100 episodes)')
    #manually set limits to top and bottom of heatmap do not get cut off 
    bottom, top = s.get_ylim()
    s.set_ylim(bottom+0.5, top-0.5)

    plt.savefig(f'V-Ki_deepdive/matrix_selections_episodes{episode_range}.pdf', bbox_inches='tight')

#################################################################################



#### ITERATION version ####

plot_cooperative_selections_whereavailable(destination_folder, titles, n_runs)

#plot_cooperation_population(destination_folder, titles, n_runs, num_iter)
plot_cooperation_population_v2(destination_folder, titles, n_runs, num_iter, with_CI=True, reduced=True) #reduced=True

#plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced=True, option=None)
plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced=True, option=None, combine_CDDC=False)

#this plot only really makes sense if each player index is of a different moral type (consistently across runs)
plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, plot_defection=False)



#plot cumulative reward avg across runs - over tieragions, not episodes 
episode_plot_reward_eachplayerinpopulation(destination_folder, titles, n_runs, num_iter, population_size, which='game') #first100 = True

episode_plot_reward_eachplayerinpopulation(destination_folder, titles, n_runs, num_iter, population_size, which='intrinsic')


if True: #plotting selection values (rather than dilemma actions) 
    plot_Q_values_population(destination_folder=destination_folder, population_size=5, n_runs=n_runs, which='local', record_Qvalues=record_Qvalues, fewer_than_nruns=False) #, fewer_than_nruns = False
    #note population_size=1 above is used to only plot the Slefish learner 
    #plot_Q_values_population(destination_folder=destination_folder, population_size=2, n_runs=n_runs, which='target') #, fewer_than_nruns = False
    #plot_Loss_population(destination_folder=destination_folder, population_size=20, n_runs=n_runs)


for run_i in range(2):
    #plot_Loss_population_new_onerun(run_idx=run_i+1, destination_folder=destination_folder, population_size=population_size, record_Loss=record_Loss)
    plot_Loss_population_newest_onerun(destination_folder=destination_folder, run_idx=run_i+1, population_size=population_size, LOSStype='dilemma')
    plot_Loss_population_newest_onerun(destination_folder=destination_folder, run_idx=run_i+1, population_size=population_size, LOSStype='selection')





#################################################################################
#### checking Q-value storage in the new way ####
os.chdir('NEWEST_IPD_DQN_PartnerSelection (statenotpairs, fullobservability) ')
destination_folder = '3xS_TEST'

destination_folder = 'results/' + destination_folder
n_runs=1
num_iterations=10000
population_size=3
titles = ['QLS','QLS','QLS']



#try plottng qvalues for one player and one run 
player_idx = 0
run_idx = 1

#opponents  = ['Selfish1', 'Selfish2']
#opponents = ['Selfish1', 'Selfish2', 'Selfish3', 'Selfish4', 'Selfish5', 'Selfish6', 'Selfish7', 'Selfish8', 'Selfish9', 'Selfish10',
#opponents  = ['Selfish1', 'Selfish2', 'Selfish3', 'Selfish4']
#             'Selfish11', 'Selfish12', 'Selfish13', 'Selfish14', 'Selfish15', 'Selfish16', 'Selfish17', 'Selfish18', 'Selfish19']

#create list of (static) opponents for a single Selfish player 
opponents_temp = long_titles.copy()
opponents_temp.remove('Selfish')
opp_indices = list(range(1, population_size))
opponents = ["{}{}".format(name, idx) for name, idx in zip(opponents_temp, opp_indices)]

Q_values_df_dil = read_new_Q_values_format(destination_folder, player_idx=player_idx, run_idx=run_idx, Qtype='dilemma')#[-100:]
plot_one_run_dilemma_Q_values_population_statenotpairs(Q_values_df_dil, run_idx=run_idx, player_idx=player_idx, maxq=5.2)#, linestyle='-', marker='.') #5, 12, 22
plot_one_run_dilemma_diff_Q_values_population_statenotpairs(Q_values_df_dil, run_idx, player_idx)

Q_values_df_sel = read_new_Q_values_format(destination_folder, player_idx=player_idx, run_idx=run_idx, Qtype='selection')#.iloc[-100:]
plot_one_run_selection_Q_values_population_statenotpairs(Q_values_df_sel, run_idx=run_idx, player_idx=player_idx, opponents=opponents, linestyle='--', marker='.') #, colors=['blue', 'cyan'])

#plot loss for all players 
plot_Loss_population_newest_onerun(destination_folder=destination_folder, run_idx=run_idx, population_size=population_size, LOSStype='dilemma')
plot_Loss_population_newest_onerun(destination_folder=destination_folder, run_idx=run_idx, population_size=population_size, LOSStype='selection')

#plot Loss for just one player 
LOSStype='dilemma'
LOSS_player = read_new_Loss_format(destination_folder, player_idx, run_idx, LOSStype)
plt.plot(LOSS_player, linewidth=0.1)
plt.title(f'{LOSStype} Loss over time, Run{run_idx}, Player{player_idx}', y=1.1)
plt.xlabel('Iteration')
plt.ylabel('Loss') 


plot_all_actions_player(destination_folder, run_idx=run_idx, player_idx=0, color='purple') # actions=['C','D'], title='Selfish'
plot_all_actions_player(destination_folder, run_idx=run_idx, player_idx=1, color='blue') # actions=['C','D'], title='Selfish'
plot_all_actions_player(destination_folder, run_idx=run_idx, player_idx=2, color='cyan') # actions=['C','D'], title='Selfish'
plot_all_actions_player(destination_folder, run_idx=run_idx, player_idx=3, color='red') # actions=['C','D'], title='Selfish'
plot_all_actions_player(destination_folder, run_idx=run_idx, player_idx=4, color='green') # actions=['C','D'], title='Selfish'


run_idx = 1 
for player_idx in range(0, population_size): 
    plot_all_actions_player(destination_folder, run_idx=run_idx, player_idx=player_idx) # , color='blue' actions=['C','D'], title='Selfish'


#plot dilemma Qvalues for each player
for player_idx in range(0,5):
    Q_values_df = read_new_Q_values_format(destination_folder, player_idx=player_idx, run_idx=run_idx, Qtype='dilemma')
    #plot_one_run_dilemma_Q_values_population_statenotpairs(Q_values_df, run_idx=run_idx, player_idx=player_idx, maxq=6) #4, 12, 22
    plot_one_run_dilemma_diff_Q_values_population_statenotpairs(Q_values_df, run_idx, player_idx)

#plot selection Qvalues for each player
for player_idx in range(0,5):
    opponents_temp = long_titles.copy()
    opponents_temp.remove('Selfish')
    opp_indices = list(range(0, population_size))
    opp_indices.remove(player_idx)
    opponents = ["{}{}".format(name, idx) for name, idx in zip(opponents_temp, opp_indices)]

    Q_values_df_sel = read_new_Q_values_format(destination_folder, player_idx=player_idx, run_idx=run_idx, Qtype='selection')
    #plot_one_run_dilemma_Q_values_population_statenotpairs(Q_values_df, run_idx=run_idx, player_idx=player_idx, maxq=6) #4, 12, 22
    plot_one_run_selection_Q_values_population_statenotpairs(Q_values_df_sel, run_idx=run_idx, player_idx=player_idx, opponents=opponents, linestyle='--', marker='.')


which = 'local'
plot_Q_values_population_statenotpairs(destination_folder, population_size, 1, which, record_Qvalues, maxq=5, fewer_than_nruns = False)
#plot_Q_values_population(destination_folder=destination_folder, population_size=5, n_runs=n_runs, which='local', record_Qvalues=record_Qvalues, fewer_than_nruns=False) #, fewer_than_nruns = False


#player0, run 1 - understsnd why it stops updating Q-values for action C after abour 5K iterations 
df = pd.read_csv(f'{destination_folder}/history/run1.csv', index_col=0)
#df[(df['idx_p1']==0) & (df['action_player1']==0)]
#df[(df['idx_p2']==0) & (df['action_player2']==0)]

#visualise early actions for a few of the static players 9GEa, GEb, GEc
player1_actions = df[df['idx_p1']==1]['action_player1'].append(df[df['idx_p2']==1]['action_player2'])
pd.DataFrame(player1_actions).sort_index(axis=0)[0:100].plot(linestyle='--', marker='o', xlabel='iteration', ylabel='action (0=C, 1=D)', title='GEa (player 1) - actions in first few iter', legend=False)

player2_actions = df[df['idx_p1']==2]['action_player1'].append(df[df['idx_p2']==2]['action_player2'])
pd.DataFrame(player2_actions).sort_index(axis=0)[0:100].plot(linestyle='--', marker='o', xlabel='iteration', ylabel='action (0=C, 1=D)', title='GEb (player 2) - actions in first few iter', legend=False)

player3_actions = df[df['idx_p1']==3]['action_player1'].append(df[df['idx_p2']==13]['action_player2'])
pd.DataFrame(player3_actions).sort_index(axis=0)[0:100].plot(linestyle='--', marker='o', xlabel='iteration', ylabel='action (0=C, 1=D)', title='GEc (player 3) - actions in first few iter', legend=False)


#### close-up Q-values - do Q(C) and Q(D) get updated at the same time? (they shouldn't) --> NO

checking = False 
if checking: 
    #### checking what happens if we do not freeze the network 
    titles = ['QLS', 'AC']
    n_runs = 20

    os.chdir('_notargetnet')
    destination_folder = '1xS_1xAC__'
    destination_folder = '1xS_1xAD__'
    destination_folder = '1xS_1xAC_1xAD__'

    os.chdir('../_updateevery1')
    destination_folder = '1xS_1xAC__'
    destination_folder = '1xS_1xAD__'
    destination_folder = '1xS_1xAC_1xAD__'

    os.chdir('../_staticsmallchecks')
    destination_folder = '1xS_2xAC_2xAD__'
    destination_folder = '1xS_1xAC_5xAD__'


    os.chdir('_bc3_with0')
    destination_folder = '5xS___iter500000_runs10'
    n_runs = 7
    num_iter = 500000
    opponents  = ['Selfish1', 'Selfish2', 'Selfish3', 'Selfish4']




    #### plot rewards as well to analyse if Qlearning is working correctly 
    df = pd.read_csv(f'{destination_folder}/history/run1.csv', index_col=0)
    df = pd.read_csv(f'{destination_folder}/history/run2.csv', index_col=0)

    selected_S1 = df[(df['idx_p1']==0) & (df['selection_player1']==0)] #from the point of view of the selecting player0, opponents Selfish1 is indexed as 0 

    selected_S2 = df[(df['idx_p1']==0) & (df['selection_player1']==1)]

    selected_S1['cumul_reward_selS1'] = np.cumsum(selected_S1['reward_game_player1'])
    selected_S2['cumul_reward_selS2'] = np.cumsum(selected_S2['reward_game_player1'])

    gamma = 0.99
    selected_S1['disc_cumul_reward_selS1'] = np.cumsum(selected_S1['reward_game_player1'] * gamma) #selected_S1['cumul_reward_selS1']*gamma
    selected_S2['disc_cumul_reward_selS2'] = selected_S2['cumul_reward_selS2']*gamma

    selection_rewards = pd.DataFrame({'episode': range(1,30001), 'select S1':selected_S1['cumul_reward_selS1'], 'select S2':selected_S2['cumul_reward_selS2']})
    selection_rewards.plot(x='episode', y=['select S1', 'select S2'], figsize=(5,4), color=['blue', 'cyan'], 
                            ylabel='cumulative reward', xlabel='episode', title='Cumulative reward for selecting Selfish1 or Selfish2', legend=True)


    disc_selection_rewards = pd.DataFrame({'episode': range(1,30001), 'select S1':selected_S1['disc_cumul_reward_selS1'], 'select S2':selected_S2['disc_cumul_reward_selS2']})
    disc_selection_rewards.plot(x='episode', y=['select S1', 'select S2'], figsize=(5,4), color=['blue', 'cyan'], 
                            ylabel='discounted \n cumulative reward', xlabel='episode', title='Discounted cumul. reward for selecting Selfish1 or Selfish2', legend=True)

    #possible Qvalue = sum_from0toK (gamma^K * reward ); K=20000 --> 5.056988325166235e-88 * r 
    np.sum(selected_S1['reward_game_player1']) * gamma**20000

    selection_rewards['disc_R_select S1'] = selection_rewards['select S1'] * (gamma ** (selection_rewards['episode'] - 1))
    selection_rewards['disc_R_select S2'] = selection_rewards['select S2'] * (gamma ** (selection_rewards['episode'] - 1))

    selection_rewards['cumul. disc. R select s1'] = selection_rewards['disc_R_select S1'].cumsum()
    selection_rewards['cumul. disc. R select s2'] = selection_rewards['disc_R_select S2'].cumsum()

    selection_rewards.plot(x='episode', y=['cumul. disc. R select s1', 'cumul. disc. R select s2'], figsize=(5,4), color=['blue', 'cyan'], 
                            ylabel='discounted \n cumulative reward', xlabel='episode', title='Disc. cumul. reward for selecting Selfish1 or Selfish2', legend=True)

    #how many times was each opponent seleted in total ?
    counts_df = pd.DataFrame({'opponent selected':['Selfish1', 'Selfish2'], 'Count':[selection_rewards.describe()['select S1']['count'], selection_rewards.describe()['select S2']['count']]}) 
    counts_df.plot.bar(x='opponent selected', y='Count', ylabel='count', color=['blue','cyan'], legend=False, title='Opponents selected by player0 \n (total number of times)')


############################################################################
#### CHeck EpsDecay works as expected ####
############################################################################ 
df = pd.read_csv(f'{destination_folder}/history/run1.csv', index_col=0)
df_player0 = df.loc[(df['idx_p1']==0) | (df['idx_p2']==0)]

df_player0['eps_player1'].plot(title='Eps_dilemma values over all iterations')

df['eps_player2']

df['eps_selection_player1'].plot(title='Eps_selection values over all iterations')

df['reason_player1'].countplot()

df['reason_selection_player1']


############################################################################
#### BEFORE IMPLEMETING STATENOTPAIRS ####
previous = False 
if previous: 


    ############################################################################
    #### analyse statenotpairs ####
    ############################################################################

    os.chdir('../_statenotpairs')
    num_iter = 100000 

    game_title = 'IPD'
    record_Loss = 'both'
    record_Qvalues = 'both'


    destination_folder = '5xS___iter100000'
    n_runs=16
    population_size=5
    titles = ['QLS','QLS','QLS','QLS','QLS']

    destination_folder = '20xS___iter100000'
    n_runs = 16
    population_size=20

    os.chdir('../_statenotpairs_savingQ')
    destination_folder = '5xS__'
    n_runs = 2 
    #TO DO 

    destination_folder = '20xS___iter300000'
    num_iter = 100000 
    n_runs = 2

    ############################################################################
    #### analyse new format for storing Qvalues ####
    ############################################################################

    os.chdir('../_diffpayoffs')
    destination_folder = '5xS___iter100000'
    n_runs=15
    population_size=5
    titles = ['QLS','QLS','QLS','QLS','QLS']

    destination_folder = '3xS___iter100000'
    population_size=3
    titles = ['QLS','QLS','QLS']



    player_idx = 0 
    run_idx = 1
    record_Qval = 'both'

    #history_Qvalues_local_player = np.load(f'{destination_folder}/QVALUES/Q_VALUES_dilemma_local_player{player_idx}_list_run{run_idx}.npy', allow_pickle=True)
    with open(f'{destination_folder}/QVALUES/Q_VALUES_dilemma_local_player{player_idx}_list_run{run_idx}.pickle', 'rb') as handle:
        b = pickle.load(handle)

    print(f'Printing local Q-values, player{player_idx}, run{run_idx}:')
    if record_Qvalues == 'dilemma':
    plot_one_run_Q_values_population(Q_values_list = history_Qvalues_local_player, run_idx = run_idx, player_idx = player_idx)
    #plot_diff_one_run_Q_values(Q_values_list = history_Qvalues_local_player, run_idx = run_idx)
    if record_Qvalues == 'selection':
    plot_one_run_Q_values_selection_population(Q_values_list = history_Qvalues_local_player, run_idx = run_idx, player_idx = player_idx)

    import pickle

    a = {'hello': 'world'}

    with open('filename.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('filename.pickle', 'rb') as handle:
        b = pickle.load(handle)



    from ast import literal_eval as make_tuple
    def read_new_Q_values_format(destination_folder, player_idx, run_idx):
        q = pd.read_pickle(f'{destination_folder}/QVALUES/Q_VALUES_dilemma_local_player{player_idx}_list_run{run_idx}.pickle')
        df = pd.Series(q).reset_index(name='Qvalue_dilemma')
        df.columns=df.iloc[0] 
        df.drop(0, axis=0, inplace=True)
        df[['state', 'action', 'player_tstep']] = pd.DataFrame(df['state, action, iteration'].apply(lambda x: make_tuple(x)).tolist(), index=df.index)
        df.drop(['state, action, iteration'], axis=1, inplace=True)
        return df['Q-value dilemma']

    Q_values_series = read_new_Q_values_format(destination_folder, player_idx=0, run_idx=1)
    plot_one_run_Q_values_population(Q_values_list = Q_values_series, run_idx = 1, player_idx = 0)
    #TO DO the above 

    ############################################################################
    #### analyse diff payoffmatrix (0-1-3-4, b:c=3:1) - 11 Jul ####
    ############################################################################

    os.chdir('_diffpayoffs')


    destination_folder = '3xS___iter100000'
    num_iter = 100000 
    n_runs = 10
    population_size=3
    titles = ['QLS','QLS','QLS']
    game_title = 'IPD'
    record_Loss = 'both'
    record_Qvalues = 'both'

    destination_folder = '5xS___iter100000'
    population_size=5
    titles = ['QLS','QLS','QLS','QLS','QLS']


    os.chdir('_diffpayoffs_savingQ')
    destination_folder = '20xS__'
    n_runs = 7
    population_size=20

    Q_values_series = read_new_Q_values_format(destination_folder, player_idx=1, run_idx=2)
    plot_one_run_Q_values_population(Q_values_list = Q_values_series, run_idx = 1, player_idx = 0)

    ############################################################################
    #### analyse latest results on new cluster, fixed comma, storing Qvalues - 10 Jul ####
    ############################################################################

    os.chdir('_population3')

    destination_folder = '3xS___iter100000_runs15'

    num_iter = 100000 
    n_runs = 15
    population_size=3
    titles = ['QLS','QLS','QLS']
    game_title = 'IPD'
    record_Loss = 'dilemma'
    record_Qvalues = 'dilemma'


    plot_Q_values_population(destination_folder=destination_folder, population_size=3, n_runs=n_runs, which='local', record_Qvalues=record_Qvalues, fewer_than_nruns=False) #, fewer_than_nruns = False


    for run_i in range(n_runs):
        plot_Loss_population_new_onerun(run_idx=run_i+1, destination_folder=destination_folder, population_size=population_size, record_Loss=record_Loss)


    run1 = pd.read_csv(destination_folder+'/history/run1.csv', index_col=0)

    run1['action_player1'] = run1['action_player1'].astype(str).replace('0', 'C').str.replace('1', 'D')
    run1['action_player2'] = run1['action_player2'].astype(str).replace('0', 'C').str.replace('1', 'D')

    run1[['idx_p2', 'action_player2']][90000:100000].value_counts().plot(kind='bar')
    run1['idx_p2'].value_counts()

    actions = ['C', 'D']
    run_df = run1_last1K

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(7,2.5))
    #plt.suptitle('Last 10K actions for each player')
    for player_idx in range(population_size): 
        ax = locals()["ax"+str(player_idx)]
        ax.set_title(f'\n Player{player_idx} actions \n (last 1K)') #No of times each action was played 
        #plt.clf()
        run1_playeridx = run1[99000:100000][run1['idx_p1']==player_idx]
        sns.countplot(ax=ax, data=run1, x=run1_playeridx['action_player1'], order=actions)


    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(7,2.5))
    #plt.suptitle('Last 10K actions for each player')
    for player_idx in range(population_size): 
        ax = locals()["ax"+str(player_idx)]
        ax.set_title(f'\n Player{player_idx} actions \n (last 1K)') #No of times each action was played 
        #plt.clf()
        run1_playeridx = run1[99000:100000][run1['idx_p2']==player_idx]
        sns.countplot(ax=ax, data=run1, x=run1_playeridx['action_player2'], order=actions)
        


    run1_last1K = run1[99000:100000]#[run1['idx_p2']==player_idx]
    run1_last2K = run1[98000:100000]
    run1_last500 = run1[99500:100000]

    sns.countplot(data=run1, x=run1_last1K['idx_p2'], order=[0,1,2])
    plt.title('# times each player is selectED (last 1K iter)')

    sns.countplot(data=run1, x=run1_last1K['idx_p1'], order=[0,1,2])
    plt.title('# times each player is selectOR')


    run1_last1K[run1_last1K['idx_p1']==3][['action_player1', 'action_player2']].value_counts()
    run1_last1K[run1_last1K['idx_p1']==3]['action_player1'].value_counts()

    run1_last1K[run1_last1K['idx_p1']==3]['reward_game_player1'].sum()

    run1_last1K[['idx_p1','reward_game_player1']].groupby('idx_p1').sum().plot(kind='bar', title='Reward_game_player1 (last 1K)', legend=None)
    run1_last1K[['idx_p2','reward_game_player2']].groupby('idx_p2').sum().plot(kind='bar', title='Reward_game_player2 (last 1K)', legend=None)






    ############################################################################
    #### checking why partner selection doesn't seem to work in 5xS case ####
    ############################################################################

    destination_folder = '5xS___iter100000_runs1'
    destination_folder = 'NEW_IPD_DQN_PartnerSelection/'+'results/' + destination_folder 
    run1 = pd.read_csv(destination_folder+'/history/run1.csv', index_col=0)
    #who gets selected towards the end? --> players 3 and 0 are most popular, player 2 is least popular 
    run1['idx_p2'][90000:100000].value_counts()
    #sanity check that selectors are correct 
    run1['idx_p1'][90000:100000].value_counts()
    #reformat action columns to C and D instead of 0 and 1
    run1['action_player1'] = run1['action_player1'].astype(str).replace('0', 'C').str.replace('1', 'D')
    run1['action_player2'] = run1['action_player2'].astype(str).replace('0', 'C').str.replace('1', 'D')

    run1[['idx_p2', 'action_player2']][90000:100000].value_counts().plot(kind='bar')

    actions = ['C', 'D']
    population_size=5

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(11,2.5))
    #plt.suptitle('Last 10K actions for each player')
    for player_idx in range(population_size): 
        ax = locals()["ax"+str(player_idx)]
        ax.set_title(f'\n Player{player_idx} actions') #No of times each action was played 
        #plt.clf()
        run1_playeridx = run1[99000:100000][run1['idx_p1']==player_idx]
        sns.countplot(ax=ax, data=run1, x=run1_playeridx['action_player1'], order=actions)


    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(11,2.5))
    #plt.suptitle('Last 10K actions for each player')
    for player_idx in range(population_size): 
        ax = locals()["ax"+str(player_idx)]
        ax.set_title(f'\n Player{player_idx} actions') #No of times each action was played 
        #plt.clf()
        run1_playeridx = run1[99000:100000][run1['idx_p2']==player_idx]
        sns.countplot(ax=ax, data=run1, x=run1_playeridx['action_player2'], order=actions)
        


    run1_last1K = run1[99000:100000]#[run1['idx_p2']==player_idx]
    run1_last2K = run1[98000:100000]
    run1_last500 = run1[99500:100000]

    sns.countplot(data=run1, x=run1_last1K['idx_p2'], order=[0,1,2,3,4])
    plt.title('# times each player is selectED (last 1K iter)')

    sns.countplot(data=run1, x=run1_last1K['idx_p1'], order=[0,1,2,3,4])
    plt.title('# times each player is selectOR')


    run1_last1K[run1_last1K['idx_p1']==3][['action_player1', 'action_player2']].value_counts()
    run1_last1K[run1_last1K['idx_p1']==3]['action_player1'].value_counts()

    run1_last1K[run1_last1K['idx_p1']==3]['reward_game_player1'].sum()

    run1_last1K[['idx_p1','reward_game_player1']].groupby('idx_p1').sum().plot(kind='bar', title='Reward_game_player1', legend=None)
    run1_last1K[['idx_p2','reward_game_player2']].groupby('idx_p2').sum().plot(kind='bar', title='Reward_game_player2', legend=None)




    ############################################################################
    #### checking Q-values storage for every 100 (instead of every value) ####
    ############################################################################

    os.chdir('../NEW_IPD_DQN_PartnerSelection')
    destination_folder = '5xS_TEST_Qvalstorage'

    destination_folder = '3xS_TEST_iter20000'
    num_iter = 20000 
    n_runs = 1 
    population_size=3
    titles = ['QLS','QLS','QLS']
    game_title = 'IPD'
    record_Loss = 'selection'
    record_Qvalues = 'selection'


    destination_folder = '3xS_TEST_iter20000_storedilemma'
    record_Loss = 'dilemma'
    record_Qvalues = 'dilemma'

    destination_folder = 'results/' + destination_folder

    plot_Q_values_population(destination_folder=destination_folder, population_size=3, n_runs=n_runs, which='local', record_Qvalues=record_Qvalues, fewer_than_nruns=False) #, fewer_than_nruns = False


    for run_i in range(n_runs):
        plot_Loss_population_new_onerun(run_idx=run_i+1, destination_folder=destination_folder, population_size=population_size, record_Loss=record_Loss)


    run1 = pd.read_csv(destination_folder+'/history/run1.csv', index_col=0)
    run1_last2K = run1[18000:]

    sns.countplot(data=run1, x=run1_last2K['idx_p2'], order=[0,1,2])
    plt.title('# times each player is selectED (last 2K iter)')

    sns.countplot(data=run1, x=run1_last2K['idx_p1'], order=[0,1,2])
    plt.title('# times each player is selectOR')


    run1_last2K[run1_last2K['idx_p1']==1][['action_player1', 'action_player2']].value_counts()
    run1_last2K[run1_last2K['idx_p1']==1]['action_player1'].value_counts()

    run1_last1K[run1_last1K['idx_p1']==3]['reward_game_player1'].sum()

    run1_last2K[['idx_p1','reward_game_player1']].groupby('idx_p1').sum().plot(kind='bar', title='Reward_game_player1', legend=None)
    run1_last2K[['idx_p2','reward_game_player2']].groupby('idx_p2').sum().plot(kind='bar', title='Reward_game_player2', legend=None)

    actions = ['C', 'D']

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(11,2.5))
    #plt.suptitle('Last 10K actions for each player')
    for player_idx in range(population_size): 
        ax = locals()["ax"+str(player_idx)]
        ax.set_title(f'\n Player{player_idx} actions (last 2K)') #No of times each action was played 
        #plt.clf()
        run1_playeridx = run1_last2K[run1_last2K['idx_p1']==player_idx]
        sns.countplot(ax=ax, data=run1_playeridx, x=run1_playeridx['action_player1'], order=[0,1])


    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(11,2.5))
    #plt.suptitle('Last 10K actions for each player')
    for player_idx in range(population_size): 
        ax = locals()["ax"+str(player_idx)]
        ax.set_title(f'\n Player{player_idx} actions (last 2K)') #No of times each action was played 
        #plt.clf()
        run1_playeridx = run1_last2K[run1_last2K['idx_p2']==player_idx]
        sns.countplot(ax=ax, data=run1_playeridx, x=run1_playeridx['action_player2'], order=[0,1])
        



    ############################################################################
    #### checking RNs for each plahyer & run ####
    ############################################################################
    check_RNs = False 
    if check_RNs: 
        # player RNs" 
        #    [0] 'empty_Q': random move when dilemma memory buffer is empty, 
        #    [1] 'eps_prob': probability to compare against eps for dilemma move, 
        #    [2] 'eps_move': random dilemma move due to eps, 
        #    [3] 'static_random': move for a static agent with strategy=='random', 
        #    [4] 'random_state': initial random state to begin learning,
        #    [5] 'DQN_weights': seed for Q-network weights initialisation (INT),
        #    [6] 'memory_sampling': seed for sampling experience from memory buffer (INT) 
        #    [7] 'selection_empty_Q': random selection when selection memory buffer is empty,
        #    [8] 'selection_eps_prob': probability to compare against eps for selection move,
        #    [9] 'selection_eps_move':  random selection due to eps
        #    [10]'sel_DQN_weights': seed for selection Q-network weights initialisation (INT),
        #    [11]'sel_memory_sampling': seed for sampling selection experience from memory buffer (INT) 
        # game RMs: 
        #    [0] 'sample_leading_player': random sampling of leading_player (player1) from the populaiton as we loop over iterations 
        #    [1] 'sample_opponent': randomly sample opponent (player2) in game.random_matching (if not using partner selection)

        os.chdir('../NEW_IPD_DQN_PartnerSelection')
        destination_folder = 'results/5xS_TEST'

        os.listdir(destination_folder+'/history')

        run1_RNs = pd.read_csv(destination_folder+'/history/run1.csv', index_col=0)[['p1_RN0','p1_RN1','p1_RN2','p1_RN3','p1_RN4','p1_RN5','p1_RN6','p1_RN7','p1_RN8','p1_RN9',
                                        'p2_RN0','p2_RN1','p2_RN2','p2_RN3','p2_RN4','p2_RN5','p2_RN6','p2_RN7','p2_RN8','p2_RN9',
                                        'game_RN0','game_RN1']]
        run2_RNs = pd.read_csv(destination_folder+'/history/run2.csv', index_col=0)[['p1_RN0','p1_RN1','p1_RN2','p1_RN3','p1_RN4','p1_RN5','p1_RN6','p1_RN7','p1_RN8','p1_RN9',
                                        'p2_RN0','p2_RN1','p2_RN2','p2_RN3','p2_RN4','p2_RN5','p2_RN6','p2_RN7','p2_RN8','p2_RN9',
                                        'game_RN0','game_RN1']]

        run1_RNs.dropna(axis=1, inplace=True) #drop columns for RNs that were not used 
        run2_RNs.dropna(axis=1, inplace=True)

        run1_RNs['p1_RN0'].hist()
        run1_RNs['p1_RN5'].hist()
        run1_RNs['p1_RN6'].hist()
        run1_RNs['p1_RN7'].hist()
        run1_RNs['p1_RN9'].hist()

        run2_RNs['p1_RN0'].hist()
        run2_RNs['p1_RN5'].hist()

        # FINDING RN5 nd RN6 appear non-uniformly distributed, with a skew towards higher values (most of the time) - but they actually only get samples population_size number of times, so the skew could be spurious 


        #### are RNs correlated within a game? (ignore which player these are for)
        sns.pairplot(run1_RNs)
        sns.pairplot(run2_RNs)

        #### what about between runs ? 
        run1_RNs.set_axis([name+'_run1' for name in run1_RNs.columns], axis=1, inplace=True)
        run2_RNs.set_axis([name+'_run2' for name in run2_RNs.columns], axis=1, inplace=True)

        d = pd.concat([run1_RNs, run2_RNs], axis='columns')
        corr = d.corr() # Compute the correlation matrix
        mask = np.triu(np.ones_like(corr, dtype=bool)) # Generate a mask for the upper triangle
        f, ax = plt.subplots(figsize=(50, 42)) #(25, 20) # Set up the matplotlib figure
        cmap = sns.diverging_palette(230, 20, as_cmap=True) #generate custom diverging colormap
        sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, annot=True, vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}) # Draw the heatmap with the mask and correct aspect ratio
        # FINDING RN5 nd RN6 are correlated for different players within runs and also between runs, and correlated with game_RN0


        #### what about between different players within a run? 
        run1_RNs = pd.read_csv(destination_folder+'/history/run1.csv', index_col=0)[['idx_p1', 'idx_p2', 'p1_RN0','p1_RN1','p1_RN2','p1_RN3','p1_RN4','p1_RN5','p1_RN6','p1_RN7','p1_RN8','p1_RN9',
                                        'p2_RN0','p2_RN1','p2_RN2','p2_RN3','p2_RN4','p2_RN5','p2_RN6','p2_RN7','p2_RN8','p2_RN9',
                                        'game_RN0','game_RN1']]
        run1_RNs.dropna(axis=1, inplace=True)
        run1_RNs['idx_p1'].value_counts() #there are 5 players in total 
        idx0_p1_run1_RNs = run1_RNs[run1_RNs['idx_p1']==0].reset_index().drop('index', axis='columns') #when the player at idx 0 acts as player1 
        idx0_p2_run1_RNs = run1_RNs[run1_RNs['idx_p2']==0].reset_index().drop('index', axis='columns')
        idx1_p1_run1_RNs = run1_RNs[run1_RNs['idx_p1']==1].reset_index().drop('index', axis='columns')
        idx1_p2_run1_RNs = run1_RNs[run1_RNs['idx_p2']==1].reset_index().drop('index', axis='columns')
        idx2_p1_run1_RNs = run1_RNs[run1_RNs['idx_p1']==2].reset_index().drop('index', axis='columns')
        idx2_p2_run1_RNs = run1_RNs[run1_RNs['idx_p2']==2].reset_index().drop('index', axis='columns')
        idx3_p1_run1_RNs = run1_RNs[run1_RNs['idx_p1']==3].reset_index().drop('index', axis='columns')
        idx3_p2_run1_RNs = run1_RNs[run1_RNs['idx_p2']==3].reset_index().drop('index', axis='columns')
        idx4_p1_run1_RNs = run1_RNs[run1_RNs['idx_p1']==4].reset_index().drop('index', axis='columns')
        idx4_p2_run1_RNs = run1_RNs[run1_RNs['idx_p2']==4].reset_index().drop('index', axis='columns')

        idx0_p1_run1_RNs.set_axis([name+'_idx0_p1' for name in idx0_p1_run1_RNs.columns], axis=1, inplace=True)
        idx0_p2_run1_RNs.set_axis([name+'_idx0_p2' for name in idx0_p2_run1_RNs.columns], axis=1, inplace=True)
        idx1_p1_run1_RNs.set_axis([name+'_idx1_p1' for name in idx1_p1_run1_RNs.columns], axis=1, inplace=True)
        idx1_p2_run1_RNs.set_axis([name+'_idx1_p2' for name in idx1_p2_run1_RNs.columns], axis=1, inplace=True)
        idx2_p1_run1_RNs.set_axis([name+'_idx2_p1' for name in idx2_p1_run1_RNs.columns], axis=1, inplace=True)
        idx2_p2_run1_RNs.set_axis([name+'_idx2_p2' for name in idx2_p2_run1_RNs.columns], axis=1, inplace=True)
        idx3_p1_run1_RNs.set_axis([name+'_idx3_p1' for name in idx3_p1_run1_RNs.columns], axis=1, inplace=True)
        idx3_p2_run1_RNs.set_axis([name+'_idx3_p2' for name in idx3_p2_run1_RNs.columns], axis=1, inplace=True)
        idx4_p1_run1_RNs.set_axis([name+'_idx4_p1' for name in idx4_p1_run1_RNs.columns], axis=1, inplace=True)
        idx4_p2_run1_RNs.set_axis([name+'_idx4_p2' for name in idx4_p2_run1_RNs.columns], axis=1, inplace=True)

        #consider correlations for eg players 0 and 1 
        players_df = pd.concat([idx0_p1_run1_RNs, idx0_p2_run1_RNs, idx1_p1_run1_RNs, idx1_p2_run1_RNs, 
                                idx2_p1_run1_RNs, idx2_p2_run1_RNs, idx3_p1_run1_RNs, idx3_p2_run1_RNs, idx4_p1_run1_RNs, idx4_p2_run1_RNs
                                ], axis='columns')
        players_df = players_df.loc[:,~players_df.columns.str.startswith('idx_')] 

        #plot corr_matrix 
        corr = players_df.corr() # Compute the correlation matrix
        mask = np.triu(np.ones_like(corr, dtype=bool)) # Generate a mask for the upper triangle
        f, ax = plt.subplots(figsize=(80, 75)) #(25, 20) # Set up the matplotlib figure
        cmap = sns.diverging_palette(230, 20, as_cmap=True) #generate custom diverging colormap
        sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, annot=False, vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}) # Draw the heatmap with the mask and correct aspect ratio

        players_df['p2_RN5_idx4_p2'].value_counts()
        players_df['p1_RN5_idx4_p2'].value_counts()
        players_df['p2_RN5_idx4_p1'].value_counts()
        players_df['p1_RN5_idx4_p1'].value_counts()

        players_df['p2_RN6_idx4_p2'].value_counts()
        players_df['p1_RN6_idx4_p2'].value_counts()
        players_df['p2_RN6_idx4_p1'].value_counts()
        players_df['p1_RN6_idx4_p1'].value_counts()





    ############################################################################
    #### Initial Experiments with population_size = 5 ####
    ############################################################################


    os.chdir('NEW_IPD_DQN_PartnerSelection')
    os.chdir('_population5')

    #### set up general parameters ####
    num_iter = 100000 
    n_runs = 20
    population_size=5

    game_title = 'IPD'
    record_Loss = 'selection'
    record_Qvalues = 'dilemma'

    destination_folder = '5xS___iter100000'

    destination_folder = '5xS___iter100000_runs1'
    n_runs = 1

    destination_folder = '1xS_1xUT_1xDE_1xVEe_1xVEk___iter100000'



    ############################################################################
    #### Initial Experiments with population_size = 20 ####
    ############################################################################

    os.chdir('../_population20')

    #### set up general parameters ####
    num_iter = 100000 
    n_runs = 20
    population_size=20

    game_title = 'IPD'
    record_Loss = 'dilemma'
    record_Qvalues = 'dilemma'

    destination_folder = '20xS___iter100000'
    destination_folder = '4xS_4xUT_4xDE_4xVEe_4xVEk___iter100000'
    destination_folder = '8xS_3xUT_3xDE_3xVEe_3xVEk___iter100000'
    destination_folder = '3xS_8xUT_3xDE_3xVEe_3xVEk___iter100000'



    testing = False 
    if testing: 
        ############################################################################
        #### Parameter Search for new population experiments, fixed bugs and inefficiencies in code; 15 June 2023
        # - partner selection with DQN for IPD, batch size 1, corrected states, player2 always learns ####
        ############################################################################

        #############################################
        #### set up specific popultion size & number of iterations ####
        os.chdir('../_population5')
        population_size=5
        num_iter = 100000

        os.chdir('_population5_seed200')

        os.chdir('../_profiling')

        os.chdir('../_LR0.01')

        os.chdir('../_UPDATEEVERY100')

        os.chdir('../_UPDATEEVERY1')


        #### set up specific population  & destination folder ####

        #############################################
        destination_folder = '1xS_1xUT_1xDE_1xVEe_1xVEk___iter100000'

        titles = get_titles_for_population(destination_folder)

        opponent_titles_S=[x for x in titles if x != 'QLS']
        opponent_titles_UT=[x for x in titles if x != 'QLUT']
        opponent_titles_DE=[x for x in titles if x != 'QLDE']
        opponent_titles_VEe=[x for x in titles if x != 'QLVE_e']
        opponent_titles_VEk=[x for x in titles if x != 'QLVE_k']
        #titles = ['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k']
        #opponent_titles_S=['QLUT', 'QLDE', 'QLVE_e', 'QLVE_k']
        #opponent_titles_UT=['QLS', 'QLDE', 'QLVE_e', 'QLVE_k']
        #opponent_titles_DE=['QLS', 'QLUT', 'QLVE_e', 'QLVE_k']
        #opponent_titles_VEe=['QLS', 'QLUT', 'QLDE', 'QLVE_k']
        #opponent_titles_VEk=['QLS', 'QLUT', 'QLDE', 'QLVE_e']

        #############################################
        destination_folder = '5xS___iter100000'

        titles = get_titles_for_population(destination_folder)
        opponent_titles_S = ['QLS','QLS','QLS','QLS']

        #opponent_titles_S0=['S1','S2','S3','S4']
        #opponent_titles_S2=['S0','S1','S3','S4']
        #opponent_titles_S1=['S0','S2','S3','S4']
        #opponent_titles_S3=['S0','S1','S2','S4']
        #opponent_titles_S4=['S0','S1','S2','S3']
        #TO DO check the above makes sense 


        #############################################
        destination_folder = '5xS___iter500000'
        n_runs = 10
        num_iter = 500000

        titles = get_titles_for_population(destination_folder)


        destination_folder = '5xS___iter100000_runs1_selQval_dilemmaLoss'
        n_runs = 1
        titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS']

        #############################################
        #### set up specific popultion size & number of iterations ####
        os.chdir('_population20')
        population_size=20
        num_iter = 50000

        #############################################
        destination_folder = '20xS__'
        titles = get_titles_for_population(destination_folder)
        opponent_titles_S0=['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19']
        #opponent_titles_S1

        #############################################
        destination_folder = '4xS_4xUT_4xDE_4xVEe_4xVEk__'
        titles = get_titles_for_population(destination_folder)
        #opponent_titles_S_0=['QLS', 'QLS', 'QLS', 'QLUT', 'QLUT', 'QLUT', 'QLUT', 'QLDE', 'QLDE', 'QLDE', 'QLDE', 'QLVE_e', 'QLVE_e', 'QLVE_e', 'QLVE_e', 'QLVE_k', 'QLVE_k', 'QLVE_k', 'QLVE_k']
        opponent_titles_S=['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k']

        #opponent_titles_S_0=['QLS_1', 'QLS_2', 'QLS_3', 'QLUT_0', 'QLUT_1', 'QLUT_2', 'QLUT_3', 'QLDE_0', 'QLDE_1', 'QLDE_2', 'QLDE_3', 'QLVE_e_0', 'QLVE_e_1', 'QLVE_e_2', 'QLVE_e_3', 'QLVE_k_0', 'QLVE_k_1', 'QLVE_k_2', 'QLVE_k_3']

        #opponent_titles_S0=['S1','S2','S3','UT1','UT2','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19']
        #opponent_titles_S1


        #############################################
        #### set up specific popultion size & number of iterations ####
        os.chdir('_population20')
        population_size=20
        num_iter = 500000
        destination_folder = '20xS___iter500000'
        titles = get_titles_for_population(destination_folder)
        #opponent_titles_S0=['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19']
        n_runs = 11


#################################################################################


#################################################################################
#### customised functions for specific populaitins ####

sns.set(font_scale=2)


plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='Alternating', idx=1)
plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='GilbertElliot_typec', idx=3)


plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='Selfish', idx=0)
plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='Utilitarian', idx=1)
plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='Deontological', idx=2)
plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='VirtueEthics_equality', idx=3)
plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='VirtueEthics_kindness', idx=4)

plot_actions_player_last100(destination_folder, n_runs, actions=['C', 'D'], title='Selfish', idx=0)
plot_actions_player_last100(destination_folder, n_runs, actions=['C', 'D'], title='Utilitarian', idx=0)
plot_actions_player_last100(destination_folder, n_runs, actions=['C', 'D'], title='Deontological', idx=0)
plot_actions_player_last100(destination_folder, n_runs, actions=['C', 'D'], title='VirtueEthics_equality', idx=0)
plot_actions_player_last100(destination_folder, n_runs, actions=['C', 'D'], title='VirtueEthics_kindness', idx=0)

plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='Selfish', idx=17)
plot_actions_player_last100(destination_folder, n_runs, actions=['C', 'D'], title='Selfish', idx=17)


#color_mapping = {'Selfish':'orange', 'AlwaysCooperate':'green', 'AlwaysDefect':'orchid', 'Alternating':, 'GilbertElliot_typea':, 'GilbertElliot_typeb':, 'GilbertElliot_typec':, 'Utilitarian':'blue', 'Deontological':'turquoise', 'VirtueEthics_equality':'mediumorchid', 'VirtueEthics_kindness':'forestgreen'}

#order = ['Selfish','Utilitarian','Deontological','VirtueEthics_equality','VirtueEthics_kindness']
player_idx = 0 

order_S_titles = opponents_temp
order_S = opponents #order based on numeric index 
palette_S = [color_mapping_longtitle[t] for t in order_S_titles] #palette based on textual labels only 

#################################################################################


if population_size == 5:
    #order_S = opponents
    order_S = get_opponents_from_title(long_titles, population_size, title='Selfish', idx=0)
    order_UT = get_opponents_from_title(long_titles, population_size, title='Utilitarian', idx=1)
    order_DE = get_opponents_from_title(long_titles, population_size, title='Deontological', idx=2)
    order_VEe = get_opponents_from_title(long_titles, population_size, title='VirtueEthics_equality', idx=3)
    order_VEk = get_opponents_from_title(long_titles, population_size, title='VirtueEthics_kindness', idx=4)

    order_AL = get_opponents_from_title(long_titles, population_size, title='Altruist', idx=5)
    #order_S = ['AlwaysDefect', 'AlwaysDefect', 'AlwaysDefect', 'AlwaysCooperate']

    order_S_titles = ['Utilitarian','Deontological','VirtueEthics_equality','VirtueEthics_kindness']
    order_UT_titles = ['Selfish','Deontological','VirtueEthics_equality','VirtueEthics_kindness']
    order_DE_titles = ['Selfish','Utilitarian','VirtueEthics_equality','VirtueEthics_kindness']
    order_VEe_titles = ['Selfish','Utilitarian','Deontological','VirtueEthics_kindness']
    order_VEk_titles = ['Selfish','Utilitarian','Deontological','VirtueEthics_equality']

    palette_S = [color_mapping_longtitle[t] for t in order_S_titles]
    palette_UT = [color_mapping_longtitle[t] for t in order_UT_titles]
    palette_DE = [color_mapping_longtitle[t] for t in order_DE_titles]
    palette_VEe = [color_mapping_longtitle[t] for t in order_VEe_titles]
    palette_VEk = [color_mapping_longtitle[t] for t in order_VEk_titles]


    plot_opponents_selected(destination_folder, n_runs, palette_S, order_S, long_titles, title='Selfish', idx=0)
    plot_opponents_selected(destination_folder, n_runs, palette_UT, order_UT, long_titles, title='Utilitarian', idx=1)
    plot_opponents_selected(destination_folder, n_runs, palette_DE, order_DE, long_titles, title='Deontological', idx=2)
    plot_opponents_selected(destination_folder, n_runs, palette_VEe, order_VEe, long_titles, title='VirtueEthics_equality', idx=3)
    plot_opponents_selected(destination_folder, n_runs, palette_VEk, order_VEk, long_titles, title='VirtueEthics_kindness', idx=4)

    plot_opponents_selected_last100(destination_folder, n_runs, palette_S, order_S, long_titles, title='Selfish', idx=player_idx)
    plot_opponents_selected_last100(destination_folder, n_runs, palette_UT, order_UT, long_titles, title='Utilitarian')
    plot_opponents_selected_last100(destination_folder, n_runs, palette_DE, order_DE, long_titles, title='Deontological')
    plot_opponents_selected_last100(destination_folder, n_runs, palette_VEe, order_VEe, long_titles, title='VirtueEthics_equality')
    plot_opponents_selected_last100(destination_folder, n_runs, palette_VEk, order_VEk, long_titles, title='VirtueEthics_kindness')


    #TO DO fix the below 
    plot_opponents_selected_aggregated_last100(destination_folder, n_runs, palette_S, order_S, title='Selfish')
    plot_opponents_selected_aggregated_last100(destination_folder, n_runs, palette_UT, order_UT, title='Utilitarian')
    plot_opponents_selected_aggregated_last100(destination_folder, n_runs, palette_DE, order_DE, title='Deontological')
    plot_opponents_selected_aggregated_last100(destination_folder, n_runs, palette_VEe, order_VEe, title='VirtueEthics_equality')
    plot_opponents_selected_aggregated_last100(destination_folder, n_runs, palette_VEk, order_VEk, title='VirtueEthics_kindness')

    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[0,1000], palette=palette_S, order=order_S, title='Selfish')
    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[0,1000], palette=palette_UT, order=order_UT, title='Utilitarian')
    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[0,1000], palette=palette_DE, order=order_DE, title='Deontological')
    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[0,1000], palette=palette_VEe, order=order_VEe, title='VirtueEthics_equality')
    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[0,1000], palette=palette_VEk, order=order_VEk, title='VirtueEthics_kindness')

    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[0,2500], palette=palette_VEk, order=order_VEk, title='VirtueEthics_kindness')
    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[25000,27500], palette=palette_VEk, order=order_VEk, title='VirtueEthics_kindness')
    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[50000,52500], palette=palette_VEk, order=order_VEk, title='VirtueEthics_kindness')
    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[75000,77500], palette=palette_VEk, order=order_VEk, title='VirtueEthics_kindness')
    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=[97500,100000], palette=palette_VEk, order=order_VEk, title='VirtueEthics_kindness')

if population_size > 5: 
    #palette = ['orange', 'blue', 'turquoise', 'mediumorchid', 'forestgreen']

    #create population list from destination_folder 
    #order = long_titles 
    #order = ['Selfish','Utilitarian','Deontological','VirtueEthics_equality','VirtueEthics_kindness','Altruist']
    palette = [color_mapping_longtitle[t] for t in long_titles]

    orders = [] 
    for idx in range(population_size):
        print(idx)
        order_idx = get_opponents_from_title(long_titles, population_size, title=long_titles[idx], idx=idx)
        orders.append(order_idx)

    palettes = [] 
    for idx in range(population_size):
        palette_temp = palette.copy()
        del palette_temp[idx]
        palettes.append(palette_temp)
    
    if False: 
        order_S = order
        order_UT = order 
        order_DE = order
        order_VEe = order
        order_VEk = order
        order_AL = order 
        order_aUT = order 
        order_mDE = order
        order_Vie = order 
        order_Vagg = order
        order_aAL = order 

        palette_S = palette
        palette_UT = palette
        palette_DE = palette
        palette_VEe = palette
        palette_VEk = palette
        palette_AL = palette 
        palette_aUT = palette 
        palette_mDE = palette
        palette_Vie = palette 
        palette_Vagg = palette
        palette_aAL = palette 

#################################################################################



#plot opponentts selected by each player TYPE (aggregating aross eg multiple Selfish players, not by player index here) 
for title in OrderedSet(long_titles):
    #title = OrderedSet(long_titles)[4]
    palette_all = [color_mapping_longtitle[t] for t in long_titles]
    plot_opponents_selected_aggregated_customrange(destination_folder, n_runs, customrange=None, palette=OrderedSet(palette_all), order=OrderedSet(long_titles), title=title)



#NOTE if we do not indicate idx then it will aggregate across all players of this type 
idx=1
title='Utilitarian'
plot_opponents_selected(destination_folder, n_runs, palettes[idx], orders[idx], long_titles, title=title, idx=idx)

#custom function to analyse partners selected - in a small population
#plot_opponents_selected(destination_folder, n_runs, opponent_titles=['S_1', 'S_2', 'S_3','S_4','S_5','S_6','S_7','S_8','S_9','S_10']) #NOTE need to finish making this function adjustable 


#plot action types 
reformat_a_s_for_population(destination_folder, n_runs, population_size)
plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)

#plot selection types based on dilemma state
reformat_sel_s_for_population(destination_folder, n_runs, population_size)
plot_selection_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)


#plot selectiontypes based on prev_move of opponent 
#NOTE understand why this soesnt; work for '_larger_network_populaiton20/results/20xS___iter50000_runs10'
#reformat_sel_move_for_population(destination_folder, n_runs, population_size=1, titles=titles)  #plot for just the first player 
reformat_sel_move_for_population(destination_folder, n_runs, population_size=population_size, titles=titles) 
plot_selection_types_prevmove_area_eachplayerinpopulation(destination_folder, titles, n_runs)
#plot_selection_types_prevmove_area_eachplayerinpopulation(destination_folder=destination_folder, titles=['QLS'], n_runs=n_runs)



#plot_results(destination_folder = destination_folder, player1_title=titles[0], player2_title=titles[1], n_runs=n_runs, game_title=game_title) 
reformat_reward_for_population(destination_folder, n_runs, population_size)

plot_results_for_population(destination_folder, titles, n_runs, game_title)



#plot last 20 actions
plot_last_20_actions_eachplayerinpopulation(destination_folder, titles, n_runs)
visualise_last_20_actions_matrix_eachplayerinpopulation(destination_folder)





### deep-dive into certain runs ####

#investigate what states were seen 
run1 = pd.read_csv(f'{destination_folder}/history/run1.csv',index_col=0).iloc[-100:]
run1.head()

run1_Selfish = run1[run1['title_p1']=='Selfish']

run1_Selfish['selection_player1'].value_counts()
run1_Selfish['selection_player1'].hist(bins=20)
sns.countplot(run1_Selfish['selection_player1'])
#plt.title('selections by Selfish players')

color_mapping = {'Selfish':'orange', 'AC':'green', 'AD':'orchid', 
                 'Utilitarian':'blue', 'Deontological':'turquoise', 'VirtueEthics_equality':'mediumorchid', 'VirtueEthics_kindness':'forestgreen',
                 'anti-Utilitarian':'cyan', 'anti-Deontological':'teal', 'anti-VirtueEthics_equality':'purple', 'anti-VirtueEthics_kindness':'limegreen'}
#palette = [color_mapping[title] for title in opponent_titles] #['green','green','green','green', 'orchid','orchid','orchid','orchid','orchid']
palette = ['orange', 'blue', 'turquoise', 'mediumorchid', 'forestgreen']
order = ['Selfish','Utilitarian','Deontological','VirtueEthics_equality','VirtueEthics_kindness']

sns.set(font_scale=2)


chart = sns.countplot(run1[run1['title_p1']=='Selfish']['title_p2'], order=order, palette=palette)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
chart.set_title('Selected by Selfish players \n (last 100)')

chart = sns.countplot(run1[run1['title_p1']=='Utilitarian']['title_p2'], order=order, palette=palette)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
chart.set_title('Selected by Utilitarian players')

chart = sns.countplot(run1[run1['title_p1']=='Deontological']['title_p2'], order=order, palette=palette)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
chart.set_title('Selected by Deontological players')

chart = sns.countplot(run1[run1['title_p1']=='VirtueEthics_equality']['title_p2'], order=order, palette=palette)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
chart.set_title('Selected by VirtueEthics_equality players')

chart = sns.countplot(run1[run1['title_p1']=='VirtueEthics_kindness']['title_p2'], order=order, palette=palette)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
chart.set_title('Selected by VirtueEthics_kindness players')

for selfish_idx in range(4):
    plt.figure()
    chart = sns.countplot(run1[run1['idx_p1']==selfish_idx]['title_p2'], order=order, palette=palette) 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    chart.set_title(f'Selected by Selfish player {selfish_idx} \n (last 100)')

for UT_idx in range(4,8):
    plt.figure()
    chart = sns.countplot(run1[run1['idx_p1']==UT_idx]['title_p2'], order=order, palette=palette) 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    chart.set_title(f'Selected by Utilitarian player {UT_idx} \n (last 100)')

for DE_idx in range(8,12):
    plt.figure()
    chart = sns.countplot(run1[run1['idx_p1']==DE_idx]['title_p2'], order=order, palette=palette) 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    chart.set_title(f'Selected by Deontological player {DE_idx} \n (last 100)')

for VEe_idx in range(12,16):
    plt.figure()
    chart = sns.countplot(run1[run1['idx_p1']==VEe_idx]['title_p2'], order=order, palette=palette) 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    chart.set_title(f'Selected by Virtue-equality player {VEe_idx} \n (last 100)')

for VEk_idx in range(16,20):
    plt.figure()
    chart = sns.countplot(run1[run1['idx_p1']==VEk_idx]['title_p2'], order=order, palette=palette) 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    chart.set_title(f'Selected by Virtue-kindness player {VEk_idx} \n (last 100)')



if False: #earlier population plots, before 15th June 

    ############################################################################
    #### debug population experiments - partner selection with DQN for IPD, batch size 1, corrected states, player2 always learns ####
    ############################################################################

    os.getcwd()
    #os.chdir('../')
    os.chdir('IPD_DQN_PartnerSelection_batch1_correctstate')

    game_title = 'IPD'

    population_size=3
    n_runs = 5
    num_iter = 20000


    destination_folder = '1xS_1xAC_1xAD__5runs_20Kiter_newversion2May_withidentity'
    titles = ['QLS', 'AC', 'AD'] #set manually

    destination_folder = '3QLS__5runs_20Kiter_newversion8May'
    titles = ['QLS', 'QLS', 'QLS']
    n_runs = 4


    destination_folder = '3QLS__5runs_40Kiter_newversion8May'
    titles = ['QLS', 'QLS', 'QLS']
    n_runs = 5
    num_iter = 40000
    #os.remove(f'{destination_folder}/.DS_Store')
    #os.remove(f'{destination_folder}/history/.DS_Store')

    destination_folder = '1xS_1xAC_1xAD__5runs_40Kiter_newversion8May'
    titles = ['QLS', 'AC', 'AD']

    destination_folder = '1xS_1xAC_1xAD__15runs_50Kiter_newversion8May'
    titles = ['QLS', 'AC', 'AD']
    num_iter = 50000
    n_runs = 15



    destination_folder = '3xS___iter50000_runs15'
    titles = ['QLS', 'QLS', 'QLS']
    #TO DO


    destination_folder = '1xS_1xAC_1xAD___iter50000_runs10'
    titles = ['QLS', 'AC', 'AD']
    num_iter = 50000
    n_runs = 5


    destination_folder = '1xS_1xAC_1xAD___iter50000_runs10_spreadmatrix'
    destination_folder = '1xS_1xAC_1xAD___iter50000_runs10_LRincrease'
    destination_folder = '1xS_1xAC_1xAD___iter50000_runs10'

    titles = ['QLS', 'AC', 'AD']
    num_iter = 50000
    n_runs = 10

    destination_folder = '3xS___iter50000_runs10'
    titles = ['QLS', 'QLS', 'QLS']

    destination_folder = '10xS___iter50000_runs10'
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']
    n_runs = 10
    population_size=10


    destination_folder = '20xS___iter50000_runs10'
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']
    population_size=20

    #plot populaiton 20 with different network architecture (256 hidden nodes) 
    #os.chdir('../')
    os.chdir('_larger_network_population20')
    population_size=20
    n_runs = 9
    destination_folder = '20xS___iter50000_runs10'
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']



    #plot more baseline cases - 1 or 2 QLS players, 10 in total 
    os.chdir('_population10')
    population_size=10
    n_runs = 10
    num_iter = 50000
    destination_folder = '1xS_4xAC_5xAD___iter50000_runs10'
    titles = ['QLS', 'AC', 'AC', 'AC', 'AC', 'AD', 'AD', 'AD', 'AD', 'AD']
    opponent_titles=['AC_1', 'AC_2', 'AC_3', 'AC_4', 'AD_1', 'AD_2', 'AD_3', 'AD_4', 'AD_5']

    destination_folder = '2xS_4xAC_4xAD___iter50000_runs10'
    titles = ['QLS', 'QLS', 'AC', 'AC', 'AC', 'AC', 'AD', 'AD', 'AD', 'AD']
    opponent_titles=['QLS_2', 'AC_1', 'AC_2', 'AC_3', 'AC_4', 'AD_1', 'AD_2', 'AD_3', 'AD_4']

    destination_folder = '1xS_3xAC_3xAD_3xTFT___iter50000_runs10' #TO RUN
    destination_folder = '1xS_2xAC_2xAD_2xTFT_3xRandom___iter50000_runs10' #TO RUN



    #plot more baseline cases - 10 or 20 players, 10 in total 
    os.chdir('_larger_network_population20')
    population_size = 20 #10
    n_runs = 9
    num_iter = 50000
    destination_folder = '20xS___iter50000_runs10'
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']
    opponent_titles=['QLS_1', 'QLS_2', 'QLS_3', 'QLS_4', 'QLS_5', 'QLS_6', 'QLS_7', 'QLS_8', 'QLS_9', 'QLS_10', 'QLS_11', 'QLS_12', 'QLS_13', 'QLS_14', 'QLS_15', 'QLS_16', 'QLS_17', 'QLS_18', 'QLS_19']



    #plot more baseline cases - 1 or 2 QLS players, 10 in total 
    os.chdir('_updateevery1_larger_netowkr_population20')
    population_size = 20
    n_runs = 10
    num_iter = 100000
    destination_folder = '20xS___iter100000_runs10'
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']
    opponent_titles=['QLS_1', 'QLS_2', 'QLS_3', 'QLS_4', 'QLS_5', 'QLS_6', 'QLS_7', 'QLS_8', 'QLS_9', 'QLS_10', 'QLS_11', 'QLS_12', 'QLS_13', 'QLS_14', 'QLS_15', 'QLS_16', 'QLS_17', 'QLS_18', 'QLS_19']


    #plot  
    os.chdir('../_population20')
    population_size = 20
    n_runs = 10
    num_iter = 150000
    destination_folder = '20xS___iter150000_runs10'
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']
    opponent_titles=['QLS_1', 'QLS_2', 'QLS_3', 'QLS_4', 'QLS_5', 'QLS_6', 'QLS_7', 'QLS_8', 'QLS_9', 'QLS_10', 'QLS_11', 'QLS_12', 'QLS_13', 'QLS_14', 'QLS_15', 'QLS_16', 'QLS_17', 'QLS_18', 'QLS_19']

    #plot version where state = notpairs 
    os.chdir('../IPD_DQN_PartnerSelection_notpairs')
    population_size = 20
    n_runs = 10
    num_iter = 100000
    destination_folder = '20xS___iter100000_runs10'
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']
    opponent_titles=['QLS_1', 'QLS_2', 'QLS_3', 'QLS_4', 'QLS_5', 'QLS_6', 'QLS_7', 'QLS_8', 'QLS_9', 'QLS_10', 'QLS_11', 'QLS_12', 'QLS_13', 'QLS_14', 'QLS_15', 'QLS_16', 'QLS_17', 'QLS_18', 'QLS_19']


    #plot version where LR = 0.01 (larger) 
    os.chdir('../_LR001')
    destination_folder = '20xS___iter100000_runs10'


    #plot version where state = notpairs & updateevery=1 (not 10) 
    os.chdir('../IPD_DQN_PartnerSelection_notpairs')
    destination_folder = '20xS___iter100000_runs10_updateevery1'


    #plt baseline case that was run most recently (6 June 2023) 
    os.chdir('../IPD_DQN_PartnerSelection')
    destination_folder = '20xS___iter100000_LATEST'
    n_runs = 15
    population_size = 20
    num_iter = 100000
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']
    opponent_titles=['QLS_1', 'QLS_2', 'QLS_3', 'QLS_4', 'QLS_5', 'QLS_6', 'QLS_7', 'QLS_8', 'QLS_9', 'QLS_10', 'QLS_11', 'QLS_12', 'QLS_13', 'QLS_14', 'QLS_15', 'QLS_16', 'QLS_17', 'QLS_18', 'QLS_19']




    #### PLOTS FOR MIXED POPULATION ####
    os.chdir('../_population5')
    destination_folder = '1xS_1xUT_1xDE_1xVEe_1xVEk___iter100000'
    n_runs = 15
    population_size = 5
    num_iter = 100000
    titles = ['QLS', 'QLUT', 'QLDE', 'QLVE_e', 'QLVE_k']
    opponent_titles_S=['QLUT', 'QLDE', 'QLVE_e', 'QLVE_k']
    opponent_titles_UT=['QLS', 'QLDE', 'QLVE_e', 'QLVE_k']
    opponent_titles_DE=['QLS', 'QLUT', 'QLVE_e', 'QLVE_k']
    opponent_titles_VEe=['QLS', 'QLUT', 'QLDE', 'QLVE_k']
    opponent_titles_VEk=['QLS', 'QLUT', 'QLDE', 'QLVE_e']



    os.chdir('../_population10')
    destination_folder = '2xS_2xUT_2xDE_2xVEe_2xVEk___iter100000'
    n_runs = 15
    population_size = 10
    num_iter = 100000
    titles = ['QLS', 'QLS', 'QLUT', 'QLUT', 'QLDE', 'QLDE', 'QLVE_e', 'QLVE_e', 'QLVE_k', 'QLVE_k']
    #TO DO CHECK ORDER 

    os.chdir('../_population20')
    destination_folder = '4xS_4xUT_4xDE_4xVEe_4xVEk___iter100000'
    n_runs = 15
    population_size = 20
    num_iter = 100000
    titles = ['QLS', 'QLS','QLS', 'QLS', 'QLUT', 'QLUT', 'QLUT', 'QLUT', 'QLDE', 'QLDE', 'QLDE', 'QLDE', 'QLVE_e', 'QLVE_e', 'QLVE_e', 'QLVE_e', 'QLVE_k', 'QLVE_k', 'QLVE_k', 'QLVE_k']
    #TO DO CHECK ORDER 




    #### functions to run for each of the above destination_folders: #### 
    long_titles = [title_mapping[title] for title in titles]
    destination_folder = 'results/' + destination_folder

    reformat_a_for_population(destination_folder, n_runs, population_size)
    plot_cooperation_population(destination_folder, titles, n_runs, num_iter)

    plot_cooperative_selections_whereavailable(destination_folder, titles, n_runs)
    #plot_cooperative_selections(destination_folder, titles, n_runs, num_iter) #deep-dive into player selections 
    #plot_cooperative_selections(destination_folder, titles, n_runs, num_iter, iter_range=(0, 1000))


    plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced=True, option=None)
    plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced=False, option=None)


    plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, plot_defection= False)


    #plot_results(destination_folder = destination_folder, player1_title=titles[0], player2_title=titles[1], n_runs=n_runs, game_title=game_title) 
    reformat_reward_for_population(destination_folder, n_runs, population_size)

    plot_results_for_population(destination_folder, titles, n_runs, game_title)


    if True: #currently plotting selection values (rather than dilemma actions) 
        plot_Q_values_population(destination_folder=destination_folder, population_size=3, n_runs=n_runs, which='local') #, fewer_than_nruns = False
        #note population_size=1 above is used to only plot the Slefish learner 
        #plot_Q_values_population(destination_folder=destination_folder, population_size=2, n_runs=n_runs, which='target') #, fewer_than_nruns = False
        plot_Loss_population(destination_folder=destination_folder, population_size=20, n_runs=n_runs)

    for run_i in range(n_runs):
        plot_Loss_population_new_onerun(run_idx=run_i+1, destination_folder=destination_folder, population_size=population_size)

    #plot action types 
    reformat_a_s_for_population(destination_folder, n_runs, population_size)
    plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)

    #plot selection types based on dilemma state
    reformat_sel_s_for_population(destination_folder, n_runs, population_size)
    plot_selection_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)


    #plot selectiontypes based on prev_move of opponent 
    #NOTE understand why this soesnt; work for '_larger_network_populaiton20/results/20xS___iter50000_runs10'
    reformat_sel_move_for_population(destination_folder, n_runs, population_size=1, titles=titles)  #plot for just the first player 
    reformat_sel_move_for_population(destination_folder, n_runs, population_size=population_size, titles=titles) 
    plot_selection_types_prevmove_area_eachplayerinpopulation(destination_folder, titles, n_runs)
    #plot_selection_types_prevmove_area_eachplayerinpopulation(destination_folder=destination_folder, titles=['QLS'], n_runs=n_runs)


    plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='Selfish')
    plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='Utilitarian')
    plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='Deontological')
    plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='VirtueEthics_equality')
    plot_actions_player(destination_folder, n_runs, actions=['C', 'D'], title='VirtueEthics_kindness')


    plot_opponents_selected(destination_folder, n_runs, opponent_titles_S, title='Selfish') 
    plot_opponents_selected(destination_folder, n_runs, opponent_titles_UT, title='Utilitarian')
    plot_opponents_selected(destination_folder, n_runs, opponent_titles_DE, title='Deontological')
    plot_opponents_selected(destination_folder, n_runs, opponent_titles_VEe, title='VirtueEthics_equality')
    plot_opponents_selected(destination_folder, n_runs, opponent_titles_VEk, title='VirtueEthics_kindness')


    #custom function to analyse partners selected - in a population of 3
    plot_opponents_selected(destination_folder, n_runs, opponent_titles=['AC', 'AD']) #for population ['S','AC','AD]
    plot_opponents_selected(destination_folder, n_runs, opponent_titles=['S_1', 'S_2']) #for population ['S','S','S']
    #plot_opponents_selected(destination_folder, n_runs, opponent_titles=['S_1', 'S_2', 'S_3','S_4','S_5','S_6','S_7','S_8','S_9','S_10']) #NOTE need to finish making this function adjustable 




    #plot last 20 actions
    plot_last_20_actions_eachplayerinpopulation(destination_folder, titles, n_runs)
    visualise_last_20_actions_matrix_eachplayerinpopulation(destination_folder)



    plot_historgram_interactions(run_idx=1, num_iter=num_iter, population_size=population_size) #plot for example single run 

    for run_idx in range(1, n_runs+1): #plot fr all runs 
        plot_historgram_interactions(run_idx=run_idx, num_iter=num_iter, population_size=population_size)



    #investigate what states were seen 
    run1 = pd.read_csv(f'{destination_folder}/history/run1.csv',index_col=0)
    run1[run1['title_p1']=='Selfish']['selection_state_player1'].value_counts()





    #which opponent did the Selfish player select 
    for run in range(1,n_runs+1):
        print(f'run {run}')
        run = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)
        print(run[run['title_p1']=='Selfish']['idx_p2'].value_counts())

        plt.figure()
        run[run['title_p1']=='Selfish']['idx_p2'].plot(kind='bar') #.hist(bins=2)
        #plt.title(f'Opponents selected, run{run}')

    #which actions did the Selfish player play 
    for run in range(1,n_runs+1):
        print(f'run {run}')
        run = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)
        plt.figure()
        run[run['title_p1']=='Selfish']['action_player1'].hist(bins=2)
        #plt.title(f'Actions played, run{run}')






    ############################################################################
    #### debug population experiments - partner selection with DQN for IPD, batch size 1 ####
    ############################################################################

    os.getcwd()
    #os.chdir('../')
    os.chdir('IPD_DQN_PartnerSelection_batch1')

    game_title = 'IPD'

    n_runs = 10
    num_iter = 10000
    population_size=3

    #destination_folder = '1xS_1xAC_1xAD__10runs_10Kiter'
    #destination_folder = '1xS_1xAC_1xAD__10runs_10Kiter_forcedselectionAC'
    destination_folder = '1xS_1xAC_1xAD__10runs_10Kiter_forcedselectionAC_fixedinitialstate'
    destination_folder = '1xS_1xAC_1xAD__10runs_10Kiter_fixedinitialstate'
    n_runs = 5
    destination_folder = '1xS_1xAC_1xAD__5runs_10Kiter_fixedinitialstate_p2doesnotlearn'
    destination_folder = '1xS_1xAC_1xAD__5runs_10Kiter_fixedinitialstate_p2learns_2ddilemmastate_stateupdatedalways'


    titles = ['QLS', 'AC', 'AD'] #set manually


    n_runs = 10
    destination_folder = '3QLS__10runs_20Kiter_newversion2May'
    titles = ['QLS', 'QLS', 'QLS'] #set manually

    n_runs = 5
    destination_folder = '1xS_1xAC_1xAD__5runs_10Kiter_newversion2May_withidentity'
    titles = ['QLS', 'AC', 'AD'] #set manually


    #TO DO
    destination_folder = '3QLS__10runs_20Kiter_newversion2May_withidentity'
    titles = ['QLS', 'QLS', 'QLS'] #set manually


    long_titles = [title_mapping[title] for title in titles]
    destination_folder = 'results/' + destination_folder

    reformat_a_for_population(destination_folder, n_runs, population_size)
    plot_cooperation_population(destination_folder, titles, n_runs, num_iter)

    plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced=True, option=None)
    plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced=False, option=None)

    plot_defection = False
    plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, plot_defection)

    if False: #need player1 and player2 folders for these functions   
        plot_action_pairs_population(destination_folder = destination_folder, titles=titles, n_runs=n_runs)
        plot_action_pairs_reduced_population(destination_folder = destination_folder, titles=titles, n_runs=n_runs)


    #plot_results(destination_folder = destination_folder, player1_title=titles[0], player2_title=titles[1], n_runs=n_runs, game_title=game_title) 
    reformat_reward_for_population(destination_folder, n_runs, population_size)

    plot_results_for_population(destination_folder, titles, n_runs, game_title)


    if True: #currently not saving Q-values
        plot_Q_values_population(destination_folder=destination_folder, population_size=1, n_runs=n_runs, which='local') #, fewer_than_nruns = False
        #note population_size=1 above is used to only plot the Slefish learner 
        #plot_Q_values_population(destination_folder=destination_folder, population_size=2, n_runs=n_runs, which='target') #, fewer_than_nruns = False
        plot_Loss_population(destination_folder=destination_folder, population_size=1, n_runs=5)

    plot_Loss_population_new_onerun(run_idx=1, destination_folder=destination_folder, population_size=population_size)


    reformat_a_s_for_population(destination_folder, n_runs, population_size)
    plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)


    plot_last_20_actions_eachplayerinpopulation(destination_folder, titles, n_runs)
    visualise_last_20_actions_matrix_eachplayerinpopulation(destination_folder)


    #deep-dive into player selections 
    plot_cooperative_selections(destination_folder, titles, n_runs, num_iter) #iter_range=(0, 20000)

    plot_cooperative_selections(destination_folder, titles, n_runs, num_iter, iter_range=(0, 1000))

    plot_historgram_interactions(run_idx=1, num_iter=num_iter, population_size=population_size) #plot for example single run 

    for run_idx in range(1, n_runs+1): #plot fr all runs 
        plot_historgram_interactions(run_idx=run_idx, num_iter=num_iter, population_size=population_size)



    #investigate what states were seen 
    run1 = pd.read_csv(f'{destination_folder}/history/run1.csv',index_col=0)
    run1[run1['title_p1']=='Selfish']['selection_state_player1'].value_counts()

    for run in range(1,n_runs+1):
        print(f'run {run}')
        run = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)
        plt.figure()
        run[run['title_p1']=='Selfish']['idx_p2'].hist() #bins=2


    for run in range(1,n_runs+1):
        print(f'run {run}')
        run = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0)
        plt.figure()
        run[run['title_p1']=='Selfish']['action_player1'].hist()


    ############################################################################
    #### testing randomness - two simple runs ######
    ############################################################################
    os.chdir('IPD_DQN_PartnerSelection_withepisodes') 
    game_title = 'IPD'
    n_runs = 1
    num_iter = 5000
    population_size=4
    destination_folder = '4QLS__1run_5Kiter_TEST'
    destination_folder = '4QLS__1run_5Kiter_TEST2'
    destination_folder = '4QLS__5runs_20Kiter'
    n_runs = 5
    num_iter = 20000,
    titles = ['QLS', 'QLS', 'QLS', 'QLS']
    destination_folder = 'results/' + destination_folder
    reformat_a_for_population(destination_folder, n_runs, population_size)
    plot_cooperation_population(destination_folder, titles, n_runs, num_iter)
    plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced=True, option=None)



    ############################################################################
    #### analyse population experiments - partner selection with DQN for IPD, batch size 1 ####
    ############################################################################

    os.getcwd()
    #os.chdir('../')
    os.chdir('IPD_DQN_PartnerSelection_batch1')

    game_title = 'IPD'

    n_runs = 20
    num_iter = 20000
    population_size=5

    destination_folder = '5xS__'

    titles = get_titles_for_population(destination_folder)
    long_titles = [title_mapping[title] for title in titles]
    destination_folder = 'results/' + destination_folder

    reformat_a_for_population(destination_folder, n_runs, population_size)
    plot_cooperation_population(destination_folder, titles, n_runs, num_iter)

    plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced=True, option=None)
    plot_defection = False
    plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, plot_defection)

    if False: #need player1 and player2 folders for these functions   
        plot_action_pairs_population(destination_folder = destination_folder, titles=titles, n_runs=n_runs)
        plot_action_pairs_reduced_population(destination_folder = destination_folder, titles=titles, n_runs=n_runs)



    #plot_results(destination_folder = destination_folder, player1_title=titles[0], player2_title=titles[1], n_runs=n_runs, game_title=game_title) 
    reformat_reward_for_population(destination_folder, n_runs, population_size)

    plot_results_for_population(destination_folder, titles, n_runs, game_title)


    if False: #currently not saving Q-values
        plot_Q_values_population(destination_folder=destination_folder, population_size=6, n_runs=1, which='local') #, fewer_than_nruns = False
        #plot_Q_values_population(destination_folder=destination_folder, population_size=2, n_runs=n_runs, which='target') #, fewer_than_nruns = False
        plot_Loss_population(destination_folder=destination_folder, population_size=6, n_runs=1)


    reformat_a_s_for_population(destination_folder, n_runs, population_size)
    plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)


    plot_last_20_actions_eachplayerinpopulation(destination_folder, titles, n_runs)
    visualise_last_20_actions_matrix_eachplayerinpopulation(destination_folder)


    #deep-dive into player selections 
    plot_cooperative_selections(destination_folder, titles, n_runs, num_iter) #iter_range=(0, 20000)

    plot_historgram_interactions(run_idx=1, num_iter=num_iter, population_size=population_size)

    for run_idx in range(1, n_runs+1):
        plot_historgram_interactions(run_idx=run_idx, num_iter=num_iter, population_size=population_size)


    plot_cooperative_selections(destination_folder, titles, n_runs, num_iter, iter_range=(0, 1000))





    ############################################################################
    #### analyse population experiments - partner selection with DQN for IPD ####
    ############################################################################

    os.getcwd()
    #os.chdir('../')
    os.chdir('IPD_DQN_PartnerSelection')

    game_title = 'IPD'

    n_runs = 1
    num_iter = 20000
    population_size=6

    destination_folder = '6QLS__1run_20Kiter_simplerstate'

    num_iter = 10000
    destination_folder = '6QLS__1run_10Kiter_simplerstate'

    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']

    num_iter = 20000
    population_size=10
    destination_folder = '10QLS__1run_20Kiter_simplerstate_epsselection01'
    destination_folder = '10QLS__1run_20Kiter_simplerstate_epsselection01_saveselectionmove'
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']



    # adding implementations from the cluster 
    n_runs = 20
    destination_folder = '10xS__' #TO DO cooperative selections ?
    destination_folder = '1xS_2xUT_2xDE_2xVEe_2xVEk_1xVM__' #TO DO 

    population_size = 50
    destination_folder = '50xS__' #TO DO cooperative selections ?


    population_size = 20
    num_iter = 300000
    n_runs = 9
    destination_folder = '20xS___iter300000'


    destination_folder = '20QLS__1run_20Kiter_simplerstate_epsselection01_saveselectionmove_updatedmemoryobj_TEST'
    n_runs = 1 
    num_iter = 20000
    titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']


    titles = get_titles_for_population(destination_folder)

    long_titles = [title_mapping[title] for title in titles]
    #titles = get_titles_for_population(destination_folder)
    destination_folder = 'results/' + destination_folder


    reformat_a_for_population(destination_folder, n_runs, population_size)
    plot_cooperation_population(destination_folder, titles, n_runs, num_iter)

    plot_action_pairs_population_new(destination_folder, titles, n_runs, reduced=True, option=None)
    plot_defection = False
    plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, plot_defection)

    if False: #need player1 and player2 folders for these functions   
        plot_action_pairs_population(destination_folder = destination_folder, titles=titles, n_runs=n_runs)
        plot_action_pairs_reduced_population(destination_folder = destination_folder, titles=titles, n_runs=n_runs)



    #plot_results(destination_folder = destination_folder, player1_title=titles[0], player2_title=titles[1], n_runs=n_runs, game_title=game_title) 
    reformat_reward_for_population(destination_folder, n_runs, population_size)

    plot_results_for_population(destination_folder, titles, n_runs, game_title)


    if False: #currently not saving Q-values
        plot_Q_values_population(destination_folder=destination_folder, population_size=6, n_runs=1, which='local') #, fewer_than_nruns = False
        #plot_Q_values_population(destination_folder=destination_folder, population_size=2, n_runs=n_runs, which='target') #, fewer_than_nruns = False
        plot_Loss_population(destination_folder=destination_folder, population_size=6, n_runs=1)


    reformat_a_s_for_population(destination_folder, n_runs, population_size)
    plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)


    plot_last_20_actions_eachplayerinpopulation(destination_folder, titles, n_runs)
    visualise_last_20_actions_matrix_eachplayerinpopulation(destination_folder)


    #deep-dive into player selections 
    plot_cooperative_selections(destination_folder, titles, n_runs, num_iter) #iter_range=(0, 20000)

    plot_historgram_interactions(run_idx=1, num_iter=num_iter, population_size=population_size)

    for run_idx in range(1, n_runs+1):
        plot_historgram_interactions(run_idx=run_idx, num_iter=num_iter, population_size=population_size)



    if False: 
        ## how many players that were selected cooperated in the previous iteration? 
        temp = pd.read_csv(destination_folder+'/history/run1.csv', index_col = 0)
        temp.head()

        temp['cooperative_selection'] = (temp['selected_prev_move']==0).apply(lambda x: int(x))
        temp['cooperative_selection'].plot(linewidth=0.06)



    # analyse - ONE-OFF

    data, results_idx = analyse_cooperative_selections(destination_folder, titles, n_runs, num_iter, iter_range=(0, 10000))
    print(data.head(15))
    data.apply(pd.value_counts)

    results_idx
    
    data, results_idx = analyse_reasons_fillingbuffer(destination_folder, titles, n_runs, num_iter, iter_range=(0, 10000))
    data.apply(pd.value_counts)
    results_idx
    #plot results 
    plt.figure(dpi=80, figsize=(25, 6))
    plt.plot(results_idx.index[:], results_idx['reason_random_fillingbuffer'], label='random selections while filling buffer', color='green', linewidth=0.1)
    plt.title(f'Random selections while filling buffer')# in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([0, 100])
    plt.ylabel("% ranodm selections \n (due to non-full memory buffer) \n (over "+str(n_runs)+r" runs)")
    plt.xlabel('Iteration')
        #leg = plt.legend() # get the legend object
        #for line in leg.get_lines(): # change the line width for the legend
        #    line.set_linewidth(4.0)

        #save the reformatted data
    #if not os.path.isdir(f'{destination_folder}/plots'):
    #        os.makedirs(f'{destination_folder}/plots')

    data, results_idx = analyse_reasons_eps(destination_folder, titles, n_runs, num_iter, iter_range=(0, 10000))
    new_data = data.apply(pd.value_counts)
    new_data.loc["% True"] = new_data.loc[True] / (new_data.loc[True] + new_data.loc[False])

    #plot results 
    plt.figure(dpi=80, figsize=(25, 6))
    plt.plot(results_idx.index[:], results_idx['reason_random_eps'], label='random selections due to eps', color='green', linewidth=0.06)
    plt.title(f'Random selections due to eps')# in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([0, 100])
    plt.ylabel("% ranodm selections \n (due to eps) \n (over "+str(n_runs)+r" runs)")
    plt.xlabel('Iteration')



    ############################################################################
    #### analyse population experiments - partner selection with DQN for IPD, INEFFICIENT STATE ####
    ############################################################################

    os.getcwd()
    os.chdir('../')
    os.chdir('IPD_DQN_PartnerSelection_inefficientstate')

    game_title = 'IPD'

    n_runs = 20
    num_iter = 20000
    population_size=10

    destination_folder = '10xS__'
    #titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']
    titles = get_titles_for_population(destination_folder)
    long_titles = [title_mapping[title] for title in titles]
    destination_folder = 'results/' + destination_folder
    plot_defection = True

    reformat_a_for_population(destination_folder, n_runs, population_size)
    plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, plot_defection)
    plot_action_pairs_population_new(destination_folder, titles, n_runs, option=None)

    plot_cooperation_population(destination_folder, titles, n_runs, num_iter)
    reformat_reward_for_population(destination_folder, n_runs, population_size)

    plot_results_for_population(destination_folder, titles, n_runs, game_title)

    reformat_a_s_for_population(destination_folder, n_runs, population_size=6)

    plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)



    plot_last_20_actions_eachplayerinpopulation(destination_folder,titles, n_runs)
    visualise_last_20_actions_matrix_eachplayerinpopulation(destination_folder)


    ############################################################################
    #### analyse population experiments - random matching with DQN for IPD ####
    ############################################################################

    os.getcwd()
    os.chdir('IPD_DQN_randomlyMatchedPartners')

    game_title = 'IPD'

    n_runs = 15
    num_iter = 20000


    destination_folder = '2xS__iter20000_runs15'
    destination_folder = '2xUT__iter20000_runs15'
    destination_folder = '2xDE__iter20000_runs15'
    destination_folder = '2xVEe__iter20000_runs15'
    destination_folder = '2xVEk__iter20000_runs15'
    destination_folder = '2xVM__iter20000_runs15'



    destination_folder = '6xS__iter20000_runs15'

    destination_folder = '5xS_1xUT__iter20000_runs15'
    destination_folder = '4xS_2xUT__iter20000_runs15'
    destination_folder = '3xS_3xUT__iter20000_runs15'
    destination_folder = '2xS_4xUT__iter20000_runs15'
    destination_folder = '1xS_5xUT__iter20000_runs15'
    destination_folder = '6xUT__iter20000_runs15'

    destination_folder = '5xS_1xDE__iter20000_runs15'
    destination_folder = '4xS_2xDE__iter20000_runs15'
    destination_folder = '3xS_3xDE__iter20000_runs15'
    destination_folder = '2xS_4xDE__iter20000_runs15'
    destination_folder = '1xS_5xDE__iter20000_runs15'
    destination_folder = '6xDE__iter20000_runs15'

    destination_folder = '5xS_1xVEe__iter20000_runs15'
    destination_folder = '4xS_2xVEe__iter20000_runs15'
    destination_folder = '3xS_3xVEe__iter20000_runs15'
    destination_folder = '2xS_4xVEe__iter20000_runs15'
    destination_folder = '1xS_5xVEe__iter20000_runs15'
    destination_folder = '6xVEe__iter20000_runs15'

    destination_folder = '5xS_1xVEk__iter20000_runs15'
    destination_folder = '4xS_2xVEk__iter20000_runs15'
    destination_folder = '3xS_3xVEk__iter20000_runs15'
    destination_folder = '2xS_4xVEk__iter20000_runs15'
    destination_folder = '1xS_5xVEk__iter20000_runs15'
    destination_folder = '6xVEk__iter20000_runs15'

    destination_folder = '5xS_1xVM__iter20000_runs15'
    destination_folder = '4xS_2xVM__iter20000_runs15'
    destination_folder = '3xS_3xVM__iter20000_runs15'
    destination_folder = '2xS_4xVM__iter20000_runs15'
    destination_folder = '1xS_5xVM__iter20000_runs15'
    destination_folder = '6xVM__iter20000_runs15'


    destination_folder = '1xS_1xUT_1xDE_1xVEe_1xVEk_1xVM___iter20000_runs15'

    #################################################################################
    #### changed population size to 100 and num_runs to 20, plotting mixed population results: 

    n_runs = 20
    num_iter = 20000
    population_size=100

    destination_folder = '5xS_19xUT_19xDE_19xVEe_19xVEk_19xVM'
    destination_folder = '20xS_16xUT_16xDE_16xVEe_16xVEk_16xVM'
    destination_folder = '60xS_8xUT_8xDE_8xVEe_8xVEk_8xVM'


    num_iter = 40000

    destination_folder = '5xS_19xUT_19xDE_19xVEe_19xVEk_19xVM__iter40000'
    destination_folder = '20xS_16xUT_16xDE_16xVEe_16xVEk_16xVM__iter40000'
    destination_folder = '60xS_8xUT_8xDE_8xVEe_8xVEk_8xVM__iter40000'



    #################################################################################
    #### functions to run for every detination_folder ####

    titles = get_titles_for_population(destination_folder)

    #################################################################################

    destination_folder = 'results/' + destination_folder

    plot_defection = False

    reformat_a_for_population(destination_folder, n_runs, population_size)
    #reformat_a_for_population_6(destination_folder, n_runs)

    plot_action_pairs_population(destination_folder = destination_folder, titles=titles, n_runs=n_runs)
    plot_action_pairs_reduced_population(destination_folder = destination_folder, titles=titles, n_runs=n_runs)

    plot_cooperation_population(destination_folder, titles, n_runs, num_iter)
    plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs)

    #plot_results(destination_folder = destination_folder, player1_title=titles[0], player2_title=titles[1], n_runs=n_runs, game_title=game_title) 
    reformat_reward_for_population(destination_folder, n_runs, population_size)

    plot_results_for_population(destination_folder, titles, n_runs, game_title)




    if False: #plotting for player1 and player2, regardless of who they are from the poulation 
        plot_actions(destination_folder = destination_folder, player1_title=titles[0], player2_title=titles[1], n_runs=n_runs)
        plot_action_types_area_population(destination_folder = destination_folder, titles = titles, n_runs=n_runs)

        plot_first_20_actions(destination_folder = destination_folder, player1_title = titles[0], player2_title = titles[1], n_runs=n_runs)
        visualise_first_20_actions_matrix(destination_folder = destination_folder)
        plot_last_20_actions(destination_folder = destination_folder, player1_title = titles[0], player2_title = titles[1], n_runs=n_runs)
        visualise_last_20_actions_matrix(destination_folder = destination_folder)


    plot_Q_values_population(destination_folder=destination_folder, population_size=6, n_runs=1, which='local') #, fewer_than_nruns = False
    #plot_Q_values_population(destination_folder=destination_folder, population_size=2, n_runs=n_runs, which='target') #, fewer_than_nruns = False

    plot_Loss_population(destination_folder=destination_folder, population_size=6, n_runs=1)


    reformat_a_s_for_population(destination_folder, n_runs)

    plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)



    plot_last_20_actions_eachplayerinpopulation(destination_folder,titles, n_runs)
    visualise_last_20_actions_matrix_eachplayerinpopulation(destination_folder)





    #deep-dive into 1xS_5xUT - what effect does the single S agent have on the UT population? 
    temp = pd.read_csv('results/1xS_5xUT__iter20000_runs15/history/run1.csv', index_col = 0)
    temp.head()
    temp['title_p1'].fillna('Selfish', inplace=True)
    temp['title_p2'].fillna('Selfish', inplace=True)

    temp['title_p1'].value_counts()
    temp['title_p2'].value_counts()
    temp['']

    
    destination_folder = '1xS_5xUT__iter20000_runs15'
    destination_folder = '6xUT__iter20000_runs15'

    titles = get_titles_for_population(destination_folder)
    destination_folder = 'results/' + destination_folder

    reformat_a_s_for_population(destination_folder, n_runs)

    plot_action_types_area_eachplayerinpopulation(destination_folder, titles, n_runs)



    plot_last_20_actions_eachplayerinpopulation(destination_folder,titles, n_runs)
    visualise_last_20_actions_matrix_eachplayerinpopulation(destination_folder)



    ############################################################################
    #### testing new population implement - random matching with DQN for IPD ####
    ############################################################################

    n_runs = 1
    destination_folder = 'results/2QLS_1run'

    n_runs = 5
    destination_folder = 'results/2QLS_5runs'

    short_titles = ['QLS', 'QLS']
    long_titles = [title_mapping[title] for title in short_titles]


    n_runs = 1
    destination_folder = 'results/6QLS_1run_10kiter'


    n_runs = 5
    destination_folder = 'results/6QLS_5runs_15kiter'

    short_titles = ['QLS', 'QLS', 'QLS', 'QLS', 'QLS', 'QLS']
    long_titles = [title_mapping[title] for title in short_titles]

    if False: 
        if 'QLVE' not in destination_folder.split('/')[1]:
            short_titles = destination_folder.split('/')[1].split('_')[0:2]
        else: #manually split the 'QLVE_' types
            short_titles = destination_folder.split('/')[1][0:6], destination_folder.split('/')[1][7:]

        long_titles = [title_mapping[title] for title in short_titles]


    plot_action_pairs(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)
    plot_results(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, game_title=game_title) 

    #plot_actions(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)
    #plot_action_types_area(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
    plot_first_20_actions(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
    visualise_first_20_actions_matrix(destination_folder = destination_folder)
    plot_last_20_actions(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
    visualise_last_20_actions_matrix(destination_folder = destination_folder)

    if False: 
        plot_Loss(destination_folder=destination_folder, n_runs=n_runs, player_x=1) #, fewer_than_nruns = False)
        plot_Loss(destination_folder=destination_folder, n_runs=n_runs, player_x=2)
        plot_Loss(destination_folder=destination_folder, n_runs=n_runs, player_x=3)
        plot_Loss(destination_folder=destination_folder, n_runs=n_runs, player_x=4)
        plot_Loss(destination_folder=destination_folder, n_runs=n_runs, player_x=5)
        plot_Loss(destination_folder=destination_folder, n_runs=n_runs, player_x=6)


    #plot_Q_values(destination_folder, n_runs=1, which='local') #, fewer_than_nruns = False
    #plot_Q_values(destination_folder, n_runs, which='target') #, fewer_than_nruns = False
    #plot_diff_Q_values(destination_folder, n_runs, which='local') #, fewer_than_nruns = False
    #plot_diff_Q_values(destination_folder, n_runs, which='target') #, fewer_than_nruns = False

    plot_Q_values_population(destination_folder=destination_folder, population_size=6, n_runs=n_runs, which='local') #, fewer_than_nruns = False
    #plot_Q_values_population(destination_folder=destination_folder, population_size=2, n_runs=n_runs, which='target') #, fewer_than_nruns = False

    plot_Loss_population(destination_folder=destination_folder, population_size=6, n_runs=n_runs)





    temp = pd.read_csv('results/1xS_5xUT_iter20000_runs15/history/run1.csv', index_col = 0)
    temp.head()
    temp['idx_p2'].value_counts()
    temp['idx_p1'].value_counts()





























    
if False: ######## OLDER PLOTTING FUNCTIONS - FROM DYADIC EXPERIMENT ########## 


    ############################################
    #### set the right working directory for this game  ####
    ############################################
    #os.getcwd()
    os.chdir('..')
    #'/Users/lizakarmannaya/RL/IPD'
    #'/Users/lizakarmannaya/Documents/PhD_data/IPD'


    #set directory according to the game 
    game_title = 'IPD'

    if game_title == 'IPD':
        #os.chdir('~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/PhD_data/IPD')
        os.chdir('IPD')
    elif game_title == 'VOLUNTEER':
        #os.chdir('~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/PhD_data/VOLUNTEER')    
        os.chdir('VOLUNTEER')  
    elif game_title == 'STAGHUNT':
        #os.chdir('~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/PhD_data/STAGHUNT')    
        os.chdir('STAGHUNT')
    print(os.getcwd())


    ############################################################################################
    #### rename directories for each ecxperiment to get rid of 'eps' and 'epsdecay\ details ####
    ############################################################################################

    for dirname in os.listdir('results'):
        if '_eps01.0_epsdecay' in dirname: 
            #get rid of formatting that reflects our parameter shoice 
            os.rename('results/'+dirname, 'results/'+dirname.replace('_eps01.0_epsdecay',''))



    ############################################
    #### template for plotting all results for one pair of players ####
    ############################################

    #plot reward & actions over time - manually put in destination_folder OR use loop below
    #destination_folder = 'results/QLS_QLS'
    destination_folder = 'results/QLVE_e_QLUT'

    n_runs = 2
    if 'QLVE' not in destination_folder.split('/')[1]:
        short_titles = destination_folder.split('/')[1].split('_')[0:2]
    else: #manually split the 'QLVE_' types
        short_titles = destination_folder.split('/')[1][0:6], destination_folder.split('/')[1][7:]
    long_titles = [title_mapping[title] for title in short_titles]

    plot_results(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, game_title=game_title) 
    #plot_actions(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)
    plot_action_pairs(destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)
    plot_action_types_area(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
    plot_first_20_actions(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
    visualise_first_20_actions_matrix(destination_folder = destination_folder)
    plot_last_20_actions(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
    visualise_last_20_actions_matrix(destination_folder = destination_folder)




    ############################################
    #### plotting all results for pair of players in a loop ####
    ############################################

    #plotting ony the pairs of actions - NB the 'plots' folder must already exist 
    QLS_list = ['QLS_QLS', 'QLUT_QLS', 'QLDE_QLS', 'QLVE_e_QLS', 'QLVE_k_QLS']
    QLmoral_QLmoral_list_1 = ['QLUT_QLUT', 'QLDE_QLUT', 'QLDE_QLDE', 'QLVE_e_QLUT']
    QLmoral_QLmoral_list_2 =['QLVE_e_QLDE', 'QLVE_e_QLVE_e', 'QLVE_k_QLUT', 'QLVE_k_QLDE', 'QLVE_k_QLVE_e', 'QLVE_k_QLVE_k']
    static_list = ['QLS_AC', 'QLS_AD', 'QLS_TFT', 'QLS_Random', 'QLUT_AC', 'QLUT_AD', 'QLUT_TFT', 'QLUT_Random', 'QLDE_AC', 'QLDE_AD', 'QLDE_TFT', 'QLDE_Random', 'QLVE_e_AC', 'QLVE_e_AD', 'QLVE_e_TFT', 'QLVE_e_Random', 'QLVE_k_AC', 'QLVE_k_AD', 'QLVE_k_TFT', 'QLVE_k_Random']

    QLS_list = ['results/'+i for i in QLS_list]
    QLmoral_QLmoral_list_1 = ['results/'+i for i in QLmoral_QLmoral_list_1]
    QLmoral_QLmoral_list_2 = ['results/'+i for i in QLmoral_QLmoral_list_2]
    static_list = ['results/'+i for i in static_list]


    n_runs = 100
    for destination_folder in static_list: #paste all 4 lists here one by one
        if 'QLVE' not in destination_folder.split('/')[1]:
            short_titles = destination_folder.split('/')[1].split('_')[0:2]
        else: #manually split the 'QLVE_' types
            short_titles = destination_folder.split('/')[1][0:6], destination_folder.split('/')[1][7:]
        long_titles = [title_mapping[title] for title in short_titles]
        print('plotting results for: ', long_titles)
        plot_action_pairs(destination_folder=destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)
        
        plot_results(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, game_title=game_title) 
        plot_actions(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)
        plot_action_types_area(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
        plot_first_20_actions(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
        visualise_first_20_actions_matrix(destination_folder = destination_folder)
        plot_last_20_actions(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
        visualise_last_20_actions_matrix(destination_folder = destination_folder)



    ############################################
    #### plotting all individual results for mixed moral player (QLVM) ####
    ############################################

    #testing
    #destination_folder = 'results/QLVM_QLS'
    #n_runs = 100
    #short_titles = ['QLVM', 'QLS']
    #long_titles = [title_mapping[title] for title in short_titles]
    #plot_results(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, game_title=game_title) 

    QLVM_list = ['QLVM_QLS', 'QLVM_QLUT', 'QLVM_QLDE', 'QLVM_QLVE_e', 'QLVM_QLVE_k', 'QLVM_QLVM']
    QLVM_static_list = ['QLVM_AC', 'QLVM_AD', 'QLVM_TFT', 'QLVM_Random']

    QLVM_list = ['results/'+i for i in QLVM_list]
    QLVM_static_list = ['results/'+i for i in QLVM_static_list]

    for destination_folder in QLVM_static_list:
        if 'QLVE' not in destination_folder.split('/')[1]:
            short_titles = destination_folder.split('/')[1].split('_')[0:2]
        else: #manually split the 'QLVE_' types
            short_titles = destination_folder.split('/')[1][0:4], destination_folder.split('/')[1][5:] #NOTE indices here are different to those in the loop plotting main results! 
        long_titles = [title_mapping[title] for title in short_titles]

        plot_action_pairs(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)
        
        plot_results(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, game_title=game_title) 
        plot_actions(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)
        plot_action_types_area(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
        plot_first_20_actions(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
        visualise_first_20_actions_matrix(destination_folder = destination_folder)
        plot_last_20_actions(destination_folder = destination_folder, player1_title = short_titles[0], player2_title = short_titles[1], n_runs=n_runs)
        visualise_last_20_actions_matrix(destination_folder = destination_folder)






    ############################################
    #### template for plotting loss for one player, given n_runs (100 too many to plot) ####
    ############################################
    #destination_folder = 'results/QLS_AC_DQN_alpha0.9_DQN1layer64nodes'

    LOSS_player1 = np.load(f'{destination_folder}/LOSS_player1_list.npy', allow_pickle=True)
    #none_list = {} 
    for run_idx in range(10):
        #none_list[run_idx] = np.isnan(LOSS_player1[run_idx])
        loss_list = [i if i else float(0) for i in LOSS_player1[run_idx]] #replace None with 0 
        loss_list = [float(i) if i!='<0.0001' else float(0) for i in loss_list] #replace all str '<0.001' with 0
        plot_one_run_Loss(LOSS_list = loss_list, run_idx = run_idx)


    ############################################
    #### template for plotting learning progress (Q-Values) for one pair of players, given n_runs (100 too many to plot) ####
    ############################################
    #use destnation_folder etc. from above
    #destination_folder = 'results/QLS_AC_DQN_alpha0.01_DQN1layer64nodes'

    Q_VALUES_player1 = np.load(f'{destination_folder}/Q_VALUES_player1_list.npy', allow_pickle=True)
    for run_idx in range(10):
        plot_one_run_Q_values(Q_values_list = Q_VALUES_player1, run_idx = run_idx)

    for run_idx in range(10):
        plot_diff_Q_values(Q_values_list = Q_VALUES_player1, run_idx = run_idx)

    Q_VALUES_player2 = np.load(f'{destination_folder}/Q_VALUES_player2_list.npy', allow_pickle=True)
    for run_idx in range(1):
        plot_one_run_Q_values(Q_values_list = Q_VALUES_player2, run_idx = run_idx)





    ############################################
    #### template for plotting relative outcomes for one player type vs. other learning players ####
    ############################################

    player1_title = 'QLVM'
    n_runs = 100

    #plot cumulative reward for various pairs in one graph
    plot_relative_reward(player1_title=player1_title, n_runs=n_runs)
    plot_relative_moral_reward(player1_title=player1_title, n_runs=n_runs) 
    #plot_cumulative_reward(player1_title=player1_title, n_runs=n_runs)

    #plot types of nmutual actions between this player & other oppponent types 
    plot_relative_action_pairs(player1_title, n_runs) #NOT USED IN THE PAPER 

    #plot % cooperating in every pair - NOTE not used in paper
    plot_relative_cooperation(player1_title=player1_title, n_runs=n_runs) 

    #### plot collective reward for various pairs one one graph, to compare relative outcomes
    plot_relative_outcomes(type='collective', player1_title=player1_title, n_runs=n_runs, game_title=game_title) 
    plot_relative_outcomes(type='gini', player1_title=player1_title, n_runs=n_runs, game_title=game_title)
    plot_relative_outcomes(type='min', player1_title=player1_title, n_runs=n_runs, game_title=game_title)







    ############################################
    #### template for plotting baseline relative outcomes for one player type vs. other static players ####
    ############################################
    #NOTE this was not done!!! (NEED TO UPDATE FUNCTIONS BEFORE RUNNING) 

    plot_baseline_relative_outcomes(type='collective', player1_title=player1_title, n_runs=n_runs, game_title=game_title) 
    plot_baseline_relative_outcomes(type='gini', player1_title=player1_title, n_runs=n_runs, game_title=game_title)
    plot_baseline_relative_outcomes(type='min', player1_title=player1_title, n_runs=n_runs, game_title=game_title)

    plot_baseline_relative_reward(player1_title=player1_title, n_runs=n_runs)
    plot_baseline_relative_moral_reward(player1_title=player1_title, n_runs=n_runs)




    plot_baseline_cumulative_reward(player1_title=player1_title, n_runs=n_runs)

    plot_basleline_relative_cooperation(player1_title=player1_title, n_runs=n_runs)

    plot_relative_baseline_action_pairs(player1_title, n_runs)


    plot_stacked_relative_reward(n_runs)
    plot_stacked_relative_moral_reward(n_runs)


    ############################################
    #### do last! plotting simultaneous action pairs ####
    ############################################

    n_runs = 100

    plot_matrix_action_pairs(n_runs)
    plot_baseline_matrix_action_pairs(n_runs)

    plot_matrix_social_outcomes(n_runs)
    plot_baseline_matrix_social_outcomes(n_runs)




    ########################################################################################
    #### extra - analysis of the effect of beta in the QLVM (CVirtue-mixed) agent ####
    ########################################################################################

    #set directory according to the game 
    game_title = 'STAGHUNT'
    #os.chdir('PhD_data')
    #os.chdir('../..')
    if game_title == 'IPD':
        os.chdir('EXTRA_results_QLVM_diffbeta/IPD')
    elif game_title == 'VOLUNTEER':
        os.chdir('EXTRA_results_QLVM_diffbeta/VOLUNTEER')  
    elif game_title == 'STAGHUNT':
        os.chdir('EXTRA_results_QLVM_diffbeta/STAGHUNT')
    os.getcwd()


    #define lists of players:
    #extra_list = ['QLVM_AD_eps01.0_epsdecay', 'QLVM_AD_eps01.0_epsdecay_beta0.2', 'QLVM_AD_eps01.0_epsdecay_beta0.4', 'QLVM_AD_eps01.0_epsdecay_beta0.6', 'QLVM_AD_eps01.0_epsdecay_beta0.8', 'QLVM_AD_eps01.0_epsdecay_beta1.0']
    extra_list = ['QLVM_QLS_eps01.0_epsdecay', 'QLVM_QLS_eps01.0_epsdecay_beta0.2', 'QLVM_QLS_eps01.0_epsdecay_beta0.4', 'QLVM_QLS_eps01.0_epsdecay_beta0.6', 'QLVM_QLS_eps01.0_epsdecay_beta0.8', 'QLVM_QLS_eps01.0_epsdecay_beta1.0']

    n_runs = 100
    for destination_folder in extra_list: #paste all 4 lists here one by one
        #short_titles = 'QLVM', 'AD'
        short_titles = 'QLVM', 'QLS'
        long_titles = [title_mapping[title] for title in short_titles]
        try:
            option = destination_folder.split('_')[4]
        except: 
            option = '_beta0'
        print('plotting results for: ', long_titles, option)
        #plot_action_pairs(destination_folder=destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, option=option)
        #using reduced pairwise action plots 
        plot_action_pairs_reduced(destination_folder=destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, option=option)


    ########################################################################################
    #### extra - analysis of the effect os smaller eps and no decay on the learning of agents ####
    ########################################################################################

    os.chdir('EXTRA_results_IPD_eps0.05')
    os.getcwd()
    n_runs = 100
    folders = ['QLS_QLS', 'QLUT_QLUT', 'QLDE_QLDE', 'QLVE_e_QLVE_e', 'QLVE_k_QLVE_k']
    for destination_folder in folders: 
        if 'QLVE' not in destination_folder:
            short_titles = destination_folder.split('_')[0:2]
        else: #manually split the 'QLVE_' types
            short_titles = destination_folder[0:6], destination_folder[7:]
        long_titles = [title_mapping[title] for title in short_titles]
        print('plotting results for: ', long_titles)
        #plot_action_pairs(destination_folder=destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, option='eps0.05')
        #using reduced pairwise action plots 
        plot_action_pairs_reduced(destination_folder=destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, option='eps0.05')












    #### TESTING new indexing in Q-value updates ####
    #run these lines in the shell 
    #python3 main.py --title1 QLS --title2 QLS --num_runs 10 --extra old_code
    #python3 main.py --title1 QLS --title2 QLS --num_runs 10 --extra new_code_indexing
    #python3 main.py --title1 QLS --title2 QLS --num_runs 100 --extra new_code_indexing

    destination_folder = 'results/QLS_QLS_runs10_old_code'
    destination_folder = 'results/QLS_QLS_runs10_new_code_indexing'
    destination_folder = 'results/QLS_QLS_runs100_new_code_indexing'

    game_title='IPD'
    n_runs = 100
    short_titles = destination_folder.split('/')[1].split('_')[0:2]
    long_titles = [title_mapping[title] for title in short_titles]
    plot_results(destination_folder = destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs, game_title=game_title) 





    ########################################################################################
    #### re-plotting table for appendix - plot via excel instead of pandas dfi explore 
    ########################################################################################

    os.chdir('/Users/lizakarmannaya/Documents/PhD_data/Dyadic Interactions (independent; tabular Q learning)/IPD/')
    os.chdir('../..')
    os.getcwd()
    destination_folder = 'results/QLVE_e_QLUT'

    save_last_20_actions_matrix(destination_folder = destination_folder)
    #now color & format in excel 



    #### re-plotting pairwise actions with fewer points on the plot - to reduce file size fo the appendix 

    os.chdir('/Users/lizakarmannaya/Documents/PhD_data/Dyadic Interactions (independent; tabular Q learning)/')

    #set directory according to the game 
    game_title = 'STAGHUNT'

    if game_title == 'IPD':
        #os.chdir('~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/PhD_data/IPD')
        os.chdir('IPD')
    elif game_title == 'VOLUNTEER':
        #os.chdir('~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/PhD_data/VOLUNTEER')    
        os.chdir('VOLUNTEER')  
    elif game_title == 'STAGHUNT':
        #os.chdir('~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/PhD_data/STAGHUNT')    
        os.chdir('STAGHUNT')
    print(os.getcwd())

    ##testing
    #destination_folder = 'results/QLS_QLS'
    #n_runs = 100
    #plot_action_pairs_reduced(destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)

    #plotting ony the pairs of actions - NB the 'plots' folder must already exist 
    QLS_list = ['QLS_QLS', 'QLUT_QLS', 'QLDE_QLS', 'QLVE_e_QLS', 'QLVE_k_QLS']
    QLmoral_QLmoral_list_1 = ['QLUT_QLUT', 'QLDE_QLUT', 'QLDE_QLDE', 'QLVE_e_QLUT']
    QLmoral_QLmoral_list_2 =['QLVE_e_QLDE', 'QLVE_e_QLVE_e', 'QLVE_k_QLUT', 'QLVE_k_QLDE', 'QLVE_k_QLVE_e', 'QLVE_k_QLVE_k']
    static_list = ['QLS_AC', 'QLS_AD', 'QLS_TFT', 'QLS_Random', 'QLUT_AC', 'QLUT_AD', 'QLUT_TFT', 'QLUT_Random', 'QLDE_AC', 'QLDE_AD', 'QLDE_TFT', 'QLDE_Random', 'QLVE_e_AC', 'QLVE_e_AD', 'QLVE_e_TFT', 'QLVE_e_Random', 'QLVE_k_AC', 'QLVE_k_AD', 'QLVE_k_TFT', 'QLVE_k_Random']

    QLS_list = ['results/'+i for i in QLS_list]
    QLmoral_QLmoral_list_1 = ['results/'+i for i in QLmoral_QLmoral_list_1]
    QLmoral_QLmoral_list_2 = ['results/'+i for i in QLmoral_QLmoral_list_2]
    static_list = ['results/'+i for i in static_list]


    n_runs = 100
    for destination_folder in static_list: #paste all 4 lists here one by one
        if 'QLVE' not in destination_folder.split('/')[1]:
            short_titles = destination_folder.split('/')[1].split('_')[0:2]
        else: #manually split the 'QLVE_' types
            short_titles = destination_folder.split('/')[1][0:6], destination_folder.split('/')[1][7:]
        long_titles = [title_mapping[title] for title in short_titles]
        print('plotting results for: ', long_titles)
        plot_action_pairs_reduced(destination_folder=destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)



    QLVM_list = ['QLVM_QLS', 'QLVM_QLUT', 'QLVM_QLDE', 'QLVM_QLVE_e', 'QLVM_QLVE_k', 'QLVM_QLVM']
    QLVM_static_list = ['QLVM_AC', 'QLVM_AD', 'QLVM_TFT', 'QLVM_Random']

    QLVM_list = ['results/'+i for i in QLVM_list]
    QLVM_static_list = ['results/'+i for i in QLVM_static_list]

    for destination_folder in QLVM_static_list:
        if 'QLVE' not in destination_folder.split('/')[1]:
            short_titles = destination_folder.split('/')[1].split('_')[0:2]
        else: #manually split the 'QLVE_' types
            short_titles = destination_folder.split('/')[1][0:4], destination_folder.split('/')[1][5:] #NOTE indices here are different to those in the loop plotting main results! 
        long_titles = [title_mapping[title] for title in short_titles]
        print('plotting results for: ', long_titles)
        plot_action_pairs_reduced(destination_folder=destination_folder, player1_title=long_titles[0], player2_title=long_titles[1], n_runs=n_runs)

