
## Computer DJ

#Libraries
import pandas as pd
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import vlc
import time
from mutagen.mp3 import MP3
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import re
from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.pyplot as plt
import requests


def get_song_ids(playlist_id, sps):
    
    sp = sps['sp_generic']
    
    # Get playlist tracks
    tracks = sp.playlist_tracks(playlist_id)
    
    # Create a list to store the song IDs
    song_ids = []
    
    # Initialize the offset and initial call flag
    offset = 0
    initial_call = True
    
    # Retrieve playlist tracks using offset to get all songs
    while True:
        tracks = sp.playlist_tracks(playlist_id, offset=offset)
        items = tracks['items']
    
        # Iterate over each track in the current set of items and extract the song ID
        for item in items:
            track = item['track']
            song_id = track['id']
            song_ids.append(song_id)
    
        # Check if it's the initial call
        if initial_call:
            total_tracks = tracks['total']
            initial_call = False
    
        # Increment the offset by the maximum limit (100) for the next call
        offset += len(items)
    
        # Break the loop if all tracks have been fetched
        if offset >= total_tracks:
            break
        
    return song_ids


def build_song_df(mp3_dir, all_song_ids, sps, comp_dj_dir):
    
    sp = sps['sp_generic']
    
    song_df_path = comp_dj_dir + 'songs_df.csv'
    try:
        songs_df = pd.read_csv(song_df_path )
        starting_index = len(songs_df)
        print("Using existing songs_df")
        print("Starting at row: {}".format(starting_index))
    except:
        print("No songs_df found, starting from scratch")
        starting_index = 0 
    
    # Get a list of all files with ".mp3" extension in the directory
    file_list = [file for file in os.listdir(mp3_dir) if file.endswith('.mp3')]

    # Create a list of song names without the ".mp3" extension
    songs = [os.path.splitext(file)[0] for file in file_list]

    # Create an empty dataframe with a column named "file name"
    song_dict_list = []
    counter = 0
    # Iterate over each song and search for matching tracks on Spotify
    print("Creating song dataframe...")
    
    for song in songs[starting_index:]:
        counter += 1
        print("{} of {}".format(counter, len(songs)))
        
        if counter%10 ==0 :
            print("Saving intermediate df...")
            # Create a list of dataframes from the dictionaries
            dfs = [pd.DataFrame([dictionary]) for dictionary in song_dict_list]
            # Concatenate the dataframes into a single dataframe
            songs_df = pd.concat(dfs, ignore_index=True)
            songs_df.to_csv(song_df_path, index=False)
        
        
        match_index = -1
        
        # Search for tracks using the song name
        try:
            song_path = mp3_dir + song + '.mp3'
            mp3_file = MP3(song_path)
            artist = mp3_file['TPE1'].text[0]
            artist = artist.split(',')[0]
            artist = artist.split('feat')[0]
            artist = re.sub(r'[^a-zA-Z ]+', '', artist)
            title = mp3_file['TIT2'].text[0]
            results = sp.search(q="artist:" + artist + " track:" + title, limit=20)
            items = results['tracks']['items']
            returned_ids = []
            for item in items:
                returned_ids.append( item['id'])
            match_index = -1
            for index, id in enumerate(returned_ids):
                if id in all_song_ids:
                    match_index = index
        
        except:
            print("Meta data approach failed for: {}".format(song))
            results = sp.search(q=song, type='track', limit=20)
            items = results['tracks']['items']
            returned_ids = []
            for item in items:
                returned_ids.append( item['id'])

            for index, id in enumerate(returned_ids):
                if id in all_song_ids:
                    match_index = index
        
        if match_index != -1:
            try:
                correct_song = items[match_index]
            
                # Retrieve track information using the Spotify API
                track_baseline = sp.track(correct_song['id'])
                track_baseline['year'] = int( track_baseline['album']['release_date'][0:4] )
                track_baseline['album'] = track_baseline['album']['name']
                track_baseline['artists'] = track_baseline['artists'][0]['name']
                vars_to_keep = ['id', 'name', 'artists', 'album',  'year', 'popularity']
                track_baseline = {key: track_baseline[key] for key in vars_to_keep if key in track_baseline}
                
                track_features = sp.audio_features(correct_song['id'])[0]
                
                vars_to_keep = ['danceability', 'energy', 'key', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',  'duration_m']
                track_features['duration_m'] = round( (track_features['duration_ms']/1000)/60, 4)
                
                track_features =  {key: track_features[key] for key in vars_to_keep if key in track_features}
    
                track_dict = {**track_baseline, **track_features}
                track_dict['file name'] = song
                
                # Append the row to the dataframe
                song_dict_list.append(track_dict)
                
            except:
                print("Couldn't get song data for song: {}".format(song))
    
    # Create a list of dataframes from the dictionaries
    dfs = [pd.DataFrame([dictionary]) for dictionary in song_dict_list]

    # Concatenate the dataframes into a single dataframe
    songs_df = pd.concat(dfs, ignore_index=True)

    return songs_df


def build_song_df_not_mp3(all_song_ids, sps):
    

    sp = sps['sp_generic']
    song_dict_list = []
    counter = 0
    # Iterate over each song and search for matching tracks on Spotify
    print("Creating song dataframe...")
    
    for song_id in all_song_ids:
        counter += 1
        print("{} of {}".format(counter, len(all_song_ids)))
        
        try:
            # Retrieve track information using the Spotify API
            track_baseline = sp.track(song_id )
            track_baseline['year'] = int( track_baseline['album']['release_date'][0:4] )
            track_baseline['album'] = track_baseline['album']['name']
            track_baseline['artists'] = track_baseline['artists'][0]['name']
            vars_to_keep = ['id', 'uri', 'name', 'artists', 'album',  'year', 'popularity']
            track_baseline = {key: track_baseline[key] for key in vars_to_keep if key in track_baseline}
            
            track_features = sp.audio_features(song_id)[0]
            
            vars_to_keep = ['danceability', 'energy', 'key', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',  'duration_m']
            track_features['duration_m'] = round( (track_features['duration_ms']/1000)/60, 4)
            
            track_features =  {key: track_features[key] for key in vars_to_keep if key in track_features}
    
            track_dict = {**track_baseline, **track_features}
            
            
            # Append the row to the dataframe
            song_dict_list.append(track_dict)
            
        except:
                print("Couldn't get song data for song: {}".format(song_id))
    
    # Create a list of dataframes from the dictionaries
    dfs = [pd.DataFrame([dictionary]) for dictionary in song_dict_list]

    # Concatenate the dataframes into a single dataframe
    songs_df = pd.concat(dfs, ignore_index=True)

    return songs_df



def load_song( player, song_path):
    
    media = vlc.Media(song_path)
    player.set_media(media)
    
    # Get the duration of the media in milliseconds
    mutagen_mp3 = MP3(song_path)
    duration_sec = int( mutagen_mp3.info.length )
    
    
    return (player, duration_sec)
    
  
def determine_next_song_index(songs_df, curr_index, already_played_indices, options, random_level = 1):
    
    vars_for_sim = songs_df[ ['popularity', 'year', 'danceability',
           'energy', 'key', 'speechiness', 'acousticness', 'instrumentalness',
           'liveness', 'valence', 'tempo' ]  ] 
    
    # Standardize the variables
    scaler = StandardScaler()
    df_standardized = pd.DataFrame( scaler.fit_transform(vars_for_sim), columns = vars_for_sim.columns)
    
    #weights
    weights = return_weights( options['weights'])
    
    for key, value in weights.items():
        df_standardized[key] = df_standardized[key] * value
    
    
    # Compute distances between observations
    distances = euclidean_distances(df_standardized)
    distances_to_curr = distances[curr_index, :]
    
    # Aritifically inflate distances for those songs that have been played
    weighter_played = np.array(songs_df['played']) * 100000000000 +1
    distances_to_curr *= weighter_played

    #depending on random level, pick either first, second, third, etc closest song
    index = random.randint(1, random_level)
    closest_index =  np.argsort(distances_to_curr)[index]
    
    counter = 2
    while closest_index ==curr_index or closest_index in already_played_indices:
        closest_index =  np.argsort(distances_to_curr)[counter]
        counter+=1

    return closest_index


def return_weights( weights_name):

    weights = {}
    weights['dj_weights'] = {
        'popularity': 5,
        'danceability': 5,
        'energy': 6,
        'key': 5,
        'speechiness': 3,
        'acousticness': 5,
        'instrumentalness': 3,
        'liveness': 3,
        'valence': 2,
        'tempo': 20,
        'year': 4
        }
    
    weights['listen_weights'] = {
        'popularity': 7,
        'danceability': 5,
        'energy': 8,
        'key': 2,
        'speechiness': 3,
        'acousticness': 6,
        'instrumentalness': 6,
        'liveness': 2,
        'valence': 3,
        'tempo': 15,
        'year': 6
        }    
    
    weights['listen_weights2'] = {
        'popularity': 8,
        'danceability': 5,
        'energy': 10,
        'key': 0,
        'speechiness': 3,
        'acousticness': 6,
        'instrumentalness': 5,
        'liveness': 2,
        'valence': 4,
        'tempo': 10,
        'year': 6
        }    
       
    try: 
        return weights[weights_name]
    except:
        print("Invalid weights name passed, choose from:\n{}\n\nReturning DJ Weights".format( weights.keys() ) )
        return(weights['dj_weights'])

        

def graph_song_network(songs_df, curr_index, next_song_index, already_played_indices, options):
    
    songs_df_id = songs_df[['id']]
    songs_df_data = songs_df.drop(['id', 'name', 'artists', 'album', 'file name' ],  axis=1)
    
    # Standardize the variables
    scaler = StandardScaler()
    songs_df_data = pd.DataFrame( scaler.fit_transform(songs_df_data), columns = songs_df_data.columns)
    
    #weights
    weights = return_weights(  options['weights'] )
    
    for key, value in weights.items():
        songs_df_data[key] = songs_df_data[key] * value
        
    pca = PCA(n_components=2)
    songs_pca =  pd.DataFrame(  pca.fit_transform(songs_df_data), columns=['PC1', 'PC2'])
    songs_pca['id'] = songs_df_id
    songs_pca['played'] = False
    songs_pca['play_next'] = False
    

    # Update 'played' column in the dataframe
    songs_pca.loc[already_played_indices, 'played'] = True
    songs_pca.loc[next_song_index, 'play_next'] = True
    
    
    plt.close('all')
    # Enable interactive mode
    plt.ion()

    plt.figure() 
    
    plt.scatter(songs_pca.loc[~songs_pca['played'], 'PC1'],
                songs_pca.loc[~songs_pca['played'], 'PC2'],
                c='red', label='Not Played', s=10)

    # Plot the observations with different colors based on the "played" column
    plt.scatter(songs_pca.loc[songs_pca['played'], 'PC1'],
                songs_pca.loc[songs_pca['played'], 'PC2'],
                c='blue', label='Played', s=60)
 
    # Plot the observations with different colors based on the "played" column
    plt.scatter(songs_pca.loc[songs_pca['play_next'], 'PC1'],
                songs_pca.loc[songs_pca['play_next'], 'PC2'],
                c='Green', label='Play Next', s=80)
    
    
    # Add "just played" text
    row_to_label = songs_pca.loc[curr_index ]
    jp_x, jp_y = [row_to_label['PC1'], row_to_label['PC2'] ]
    jp_text = "Just played: {}".format( songs_df.loc[curr_index]['file name'])

    # Add text "just played" in the top left
    plt.text(0.05, 0.95, jp_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left',
             fontsize=8, color='blue')

    # Add "about to play" text
    row_to_label = songs_pca.loc[next_song_index ]
    atp_x, atp_y = [ row_to_label['PC1'], row_to_label['PC2'] ]    
    atp_text = "About to play: {}".format( songs_df.loc[next_song_index]['file name'])   

    # Add text "just played" in the top left
    plt.text(0.05, 0.9, atp_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left',
             fontsize=8, color='green')
    
    arrow_properties = dict(arrowstyle='->', color='black', linewidth=2.5, mutation_scale=20)
    plt.annotate('', xy=(atp_x, atp_y), xytext=(jp_x, jp_y),
             arrowprops=arrow_properties)
    

    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    
    
    return plt
    #plt.show()
    #time.sleep(1)
    #plt.show(block=False)
    


 
def song_player(songs_df, mp3_dir, options):
    
     test_mode = False
     #test_mode = True 
     
     songs_df['played'] = False
     already_played_indices = [] 
     
     # Create a VLC media player instance
     player = vlc.MediaPlayer()

     #Get a set of random songs to cycle through 
     songs_random_indices = random.sample( range(len(songs_df)), len(songs_df))
     random_counter=0
     
     #Initialize the first song as the first random song
     curr_index = songs_random_indices[random_counter]
     random_counter += 1
     
     #Allow user to cycle between first song seed
     random_input_usr = '0'
     while random_input_usr !='1':
         random_input_usr = input("First song is: {} by {}\n1.Okay\n2.Next song\n".format(songs_df['name'][curr_index] , songs_df['artists'][curr_index] ))
         if random_input_usr != "1":
             curr_index = songs_random_indices[random_counter]
             random_counter += 1
         
     #Default start
     user_input='1'
     automatic_mode = False
     automatic_num=4
     
     while user_input != '4':
         
             
         if user_input == '1':       
             
             #Allow user to pick how many next songs to play
             if not automatic_mode:
                 num_songs = input("How many songs?\n")
                 while num_songs not in [str(i) for i in range(1000)]:
                     num_songs = input("Enter number between 1 and 999\n")
                 num_songs = int(num_songs)
                 if num_songs ==0:
                     num_songs+=1
             else:
                 num_songs = automatic_num
                 
             song_counter = 0 
             
             #Play through number of "next closest" songs
             while song_counter <num_songs:
                 song_counter += 1
                 
                 # Load the MP3 file
                 fp_song = mp3_dir + songs_df['file name'][curr_index] + '.mp3'
                 player, duration_sec = load_song(player, fp_song )
                 if test_mode:
                     real_duration_sec = duration_sec
                     duration_sec = 5
                 start_time = time.time()
                 elapsed_time = 0
                 
                 
                 txt = "Now playing: {} by {}...".format(songs_df['name'][curr_index] , songs_df['artists'][curr_index] )
                 print( txt)


                 #Play song in middle of song for 5 seconds
                 if test_mode:
                     player.play(  )                     
                     player.set_time(1000*int(real_duration_sec/2)) 

                 else:
                    player.play()
                 
                 
                 #Get next closest song using distance metric
                 next_song_index = determine_next_song_index(songs_df, curr_index, already_played_indices, options)
                 
                 #Play song
                 while elapsed_time < duration_sec-3:
                     
                     #Print out a loading bar
                     dots = 60
                     pct = int( dots* elapsed_time / duration_sec) 
                     pct_mins = dots - pct
                     elapsed_time = time.time() - start_time
                     elapsed_str = "{}:{}".format( int(elapsed_time//60), str(int(elapsed_time%60)).zfill(2)  )
                     duration_str = "{}:{}".format( int(duration_sec//60), str(int(duration_sec%60)).zfill(2)  )
                     print( "[", '.'*pct, " "*pct_mins, "]" ," ", elapsed_str, " / ", duration_str, sep="")                    
                     time.sleep(3)    
                 
                 #Add song to "already Played" list
                 already_played_indices.append(curr_index)
                 
                 #Stop playing
                 player.stop()   
                 songs_df.loc[curr_index, 'played'] = True
                                
                 
                
             
                 #In automatic mode, automatically set next seed and keep playing
                 if automatic_mode:
                     if song_counter == num_songs:
                         os.system("say 'New Seed'")
                         print("*"*50, "\n\nContinuing in automatic mode - NEW SEED...\n")
                         next_song_index = songs_random_indices[random_counter]
                         random_counter += 1
                 
                 #Plot network
                 plot = graph_song_network(songs_df, curr_index, next_song_index, already_played_indices, options)
                 plot.show()
                 plot.pause(0.5)
                 
                 #Set up loop to play next song
                 curr_index = next_song_index
                 
                 
         #New seed
         if user_input == '2':
             random_input_usr = '0'
             while random_input_usr !='1':
                 random_input_usr = input("Next seed song is: {} by {}\n1.Okay\n2.Next song\n".format(songs_df['name'][curr_index] , songs_df['artists'][curr_index] ))
                 if random_input_usr != "1":
                     curr_index = songs_random_indices[random_counter]
                     random_counter += 1

         if user_input == '3':
             automatic_mode = True
             automatic_num = 4
             user_input = '1'
             
             
         if user_input == '4':
             # Stop the player
             player.stop()
         
         if not automatic_mode:
             menu_txt = '1. Play songs\n2. New Seed\n3. Automatic Mode\n4. Quit\n'
             user_input = input(menu_txt)
             while user_input not in ['1', '2', '3', '4']:
                 user_input = input("Try Again")
         else: 
             user_input = "1"
     

        
     return player
 
def get_all_playlist_songs(sps, options):
    
    sp = sps['sp_user']

    # Get user information
    user = sp.current_user()

    song_ids = []
    playlist_list = []

    # Retrieve all public playlists of the user
    if options['which_playlist'] == 'all':
        playlists = sp.current_user_playlists(limit= options['num_playlists'])  # Increase limit if needed

        
    
        # Iterate over the playlists and extract the song IDs
        while playlists:
            for playlist in playlists['items']:
                if not playlist['public']:
                    continue  # Skip non-public playlists
                playlist_list.append( sp.playlist(playlist['id'])['name'] )    
                playlist_tracks = sp.playlist_tracks(playlist['id'], fields="items(track(id))")
                song_ids += [track['track']['id'] for track in playlist_tracks['items']]
    
            if playlists['next']:
                playlists = sp.next(playlists)
            else:
                playlists = None
                
    elif options['which_playlist'] == 'specified':
        for play_name, play_uri in options['playlist_dict'].items():
            print('getting songs for playlist: {}'.format(play_name) )
            playlist_list.append( sp.playlist(play_uri)['name'] )
            
            
            results = sp.playlist_tracks(play_uri)
            tracks = results['items']
            while results['next']:
                results = sp.next(results)
                tracks.extend(results['items'])
                
            for track in tracks:
                song_ids.append(track['track']['id'])

    song_ids = list(set(song_ids))
    return {'song_ids' : song_ids, 'playlist_list': playlist_list } 

def create_spotify_playlist( sps, options):
    
    base_desc = 'A SongFlow playlist -- {num_songs} songs from {num_playlists} playlists. --Playlists used were: {playlist_list} -- Created on: {date_txt}'   
    
    
    if options['which_playlist'] == 'rips':
        print("Getting song ids from rips playlist...")
        rips_ids = get_song_ids(options['rips_archived_id'], sps)
        playlist_name = 'rips organized ' + str( options['cycle_through_multiplier']) + ' times through'
        
    elif options['which_playlist'] == 'all' or options['which_playlist'] == 'specified':
        if options['which_playlist'] == 'all':
            print("Getting song ids from last {} playlists...".format(options['num_playlists']))
            out_dict = get_all_playlist_songs(sps, options)
            rips_ids = out_dict['song_ids'] ; playlist_list = out_dict['playlist_list']
            playlist_name = 'A SongFlow Playlist with EVERY playlist - Created ' + datetime.now().strftime("%Y-%m-%d %H:%M") 
            
        
        else:
            print("Getting song ids from chosen playlists ({} total)...".format( len(options['playlist_dict']) ) )
            out_dict = get_all_playlist_songs(sps, options)
            rips_ids = out_dict['song_ids'] ; playlist_list = out_dict['playlist_list']
            
        playlist_name = 'A SongFlow Playlist - Created ' + datetime.now().strftime("%Y-%m-%d %H:%M")  
        num_songs = len(rips_ids); num_playlists = len(playlist_list)
        playlist_list_form = '/'.join( playlist_list ) 
        weights_txt = return_weights( options['weights'])
        weights_txt = "/".join([f"{key}: {value}" for key, value in weights_txt.items()])
        date_txt =  datetime.now().strftime("%B %d, %Y")
        if len(playlist_list_form) >200:
            playlist_list_form = ''
            for p in playlist_list:
                if len(p)>20:
                    playlist_list_form += (p[0:19] + ".../")
                else:
                    playlist_list_form += p + "/"
            if len(playlist_list_form) >180:
                playlist_list_form = playlist_list_form[0:177] + "..."
        playlist_desc = base_desc.format(num_songs = num_songs, num_playlists=num_playlists,playlist_list=playlist_list_form,\
                                                date_txt=date_txt)
            
    elif options['which_playlist'] == 'other':
        print("Getting song ids from other playlist...")
        rips_ids = get_song_ids( options['other_playlist_id'] , sps)
        playlist_name = 'other playlist organized ' + str( options['cycle_through_multiplier']) + ' times through'
       
        
    #dedupe
    rips_ids = list(set(rips_ids))
    
    songs_df = build_song_df_not_mp3(rips_ids, sps)
    
    num_songs = len(songs_df)
    songs_df['played'] = False
    
    
    #get first song
    songs_random_indices = random.sample( range(len(songs_df)), len(songs_df))
        
    #Initialize the first song as the first random song
    curr_index = songs_random_indices[0]
    song_uris = []
    already_played_indices = []
    
    song_uris.append( songs_df.loc[curr_index, 'uri'] )
    already_played_indices.append(curr_index)
    songs_df.loc[curr_index, 'played'] = True
    
    while len(song_uris) < options['cycle_through_multiplier'] * num_songs:
        
        if len(already_played_indices) >  options['song_memory']:
            start_i = options['song_memory'] - len(already_played_indices)
            already_played_indices = already_played_indices[ start_i: ]
        
        print(songs_df.loc[curr_index, 'name'])
        
        look4next = True
        while look4next:
            next_candidate = determine_next_song_index(songs_df, curr_index, already_played_indices, options, random_level =options['random_level'] )
            if next_candidate not in already_played_indices:
                look4next = False
        
            song_uris.append( songs_df.loc[next_candidate, 'uri'])
            already_played_indices.append(next_candidate)
            songs_df.loc[curr_index, 'played'] = True
            curr_index = next_candidate
    
     
    # Get user information
    sp_user= sps['sp_user']
    user = sp_user.me()

    # Create a new playlist
    playlist = sp_user.user_playlist_create(user['id'], name = playlist_name, description = playlist_desc)
    playlist_id = playlist['id']
    sp_user.playlist_change_details(playlist_id, description=playlist_desc)

    # Add songs to the playlist 1 at a time
    print("Building playlist with {} songs...".format(len(song_uris)))
    #counter = 0
    #for song_uri in song_uris:
    #    counter += 1
    #    print("{} out of {}".format(counter, len(song_uris) ))
    #    song_uri = song_uri.split(':')[-1] 
    #    sp_user.playlist_add_items(playlist_id, [song_uri] )
    
    #Add songs 100 at a time (max number)    
    uris_clipped = [uri.split(':')[-1] for uri in  song_uris   ] 
    batch_size = 100
    for i in range(0, len(uris_clipped), batch_size):
        batch = uris_clipped[i:i+batch_size]
        sp_user.playlist_add_items(playlist_id, batch)
    
    #Try to change playlist description
    #sp_user.playlist_change_details(playlist_id, description=playlist_desc)


    if len(playlist_desc) >300:
        playlist_desc = playlist_desc[0:300]
    
    #Try using web API
    access_token = sp_user.auth_manager.get_access_token()['access_token']  
    # Set the API endpoint
    url = f'https://api.spotify.com/v1/playlists/{playlist_id}'   
    # Set the headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }  
    # Set the data with the new description
    data = {
        'description': playlist_desc
    }
    # Make the PUT request to update the playlist description
    response = requests.put(url, headers=headers, json=data)
    
    if response.status_code == 200:
        print("Playlist description updated successfully.")
    else:
        print("Failed to update playlist description.")


    # Return the playlist information
    return playlist     
     
def main():

    options = {}
    
    rebuild_df = True    
    rebuild_df = False
    
    #Build playlist options
    build_playlist = False

    options['which_playlist'] = 'specified'    
    options['playlist_dict'] = { 'Four Tet' : 'spotify:playlist:2uzbATYxs9V8YQi5lf89WG'}
    
    #options['which_playlist'] = 'all'
    
    options['random_level'] = 1
    options['song_memory'] = 2000
    options['cycle_through_multiplier'] = 1

    #General options
    options['weights'] = 'listen_weights'
    
    line = '-' + 48*' ' + '-\n'
    full_line = "-"*50 + '\n'
    print(full_line, line, line, '-' + ' '*18 + "Computer DJ" + ' '*19 + '-\n' , line, line, full_line, sep="")

    
    print("\n\nloading....")
    time.sleep(1)
    

    
    options['rips_archived_id'] = 'spotify:playlist:4Ra7m4IDwu9GVMq7FC51Kj'
    options['other_playlist_id'] = 'spotify:playlist:7xdBkpsoNBEt6u53woNWo2'
    
    mp3_dir = '/Users/coltonlapp/Dropbox/Personal/Music/DJ/MP3s/Spotify/'
    comp_dj_dir = '/Users/coltonlapp/Dropbox/Personal/Music/DJ/SCRIPTS/ComputerDJ/'

    client_id= 'temp'
    client_secret = 'temp'
    redirect_uri = 'https://google.com'
    scope = 'playlist-modify-public'
    
    cache_path1 = comp_dj_dir + ".spotipy.cache" 
    cache_path2 = comp_dj_dir + ".spotipy2.cache" 


    
    with open( comp_dj_dir + "credentials.txt", "r") as file:
        credentials = file.readlines()
    
    # Extract the values and assign them to variables
    client_id = credentials[0].strip().split('=')[1].strip().strip("'")
    client_secret = credentials[1].strip().split('=')[1].strip().strip("'")

    # Authenticate with Spotify API
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    
    sp_generic = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    sp_generic = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, cache_path = cache_path1))
    sp_user = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope, cache_path = cache_path2))

    sps = {'sp_generic' : sp_generic, 'sp_user' : sp_user}


    if build_playlist:
        create_spotify_playlist( sps, options)
    

    # Check if the file exists
    if rebuild_df or (not os.path.exists(comp_dj_dir + 'songs_df.csv') ):
        
        all_song_ids = get_song_ids( options['rips_archived_id'], sps)
        
        #Delete old songs_df
        if rebuild_df:
            if  os.path.exists(comp_dj_dir + 'songs_df.csv'):
                os.remove(comp_dj_dir + 'songs_df.csv')
                
        # File does not exist, so run the build_songs function
        songs_df = build_song_df(mp3_dir, all_song_ids, sps, comp_dj_dir)
        songs_df.to_csv( comp_dj_dir + 'songs_df.csv', index=False)
    else:
        # File exists, read the CSV file into a dataframe
        songs_df = pd.read_csv(comp_dj_dir + 'songs_df.csv')
        
    print("Loaded {} songs...\n".format(len(songs_df)))
    os.system("say 'Welcome to Computer DJ'")
    

        
    player = song_player(songs_df, mp3_dir, options)
    player.stop()


if __name__ == '__main__':
    main()

   
    
    