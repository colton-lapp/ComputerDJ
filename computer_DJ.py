
## Computer DJ

#Libraries
import pandas as pd
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from inputimeout import inputimeout, TimeoutOccurred
import vlc
import time
import select
import sys
from mutagen.mp3 import MP3
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np



def get_song_ids(playlist_id, sp):
    
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


def build_song_df(mp3_dir, all_song_ids, sp, comp_dj_dir):
    
    song_df_path = comp_dj_dir + 'songs_df.csv'
    try:
        curr_df = pd.read_csv(song_df_path )
        starting_index = len(curr_df)
    except:
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
            songs_df.to_csv(song_df_path, index=True)
        
        # Search for tracks using the song name
        results = sp.search(q=song, type='track', limit=20)
        items = results['tracks']['items']
        returned_ids = []
        for item in items:
            returned_ids.append( item['id'])

        
        match_index = -1
        for index, id in enumerate(returned_ids):
            if id in all_song_ids:
                match_index = index
        
        if match_index != -1:
            correct_song = items[match_index]
        
            # Retrieve track information using the Spotify API
            track_baseline = sp.track(correct_song['id'])
            track_baseline['album'] = track_baseline['album']['name']
            track_baseline['artists'] = track_baseline['artists'][0]['name']
            vars_to_keep = ['id', 'name', 'artists', 'album', 'popularity']
            track_baseline = {key: track_baseline[key] for key in vars_to_keep if key in track_baseline}
            
            track_features = sp.audio_features(correct_song['id'])[0]
            
            vars_to_keep = ['danceability', 'energy', 'key', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',  'duration_m']
            track_features['duration_m'] = round( (track_features['duration_ms']/1000)/60, 4)
            
            track_features =  {key: track_features[key] for key in vars_to_keep if key in track_features}

            track_dict = {**track_baseline, **track_features}
            track_dict['file name'] = song
            
            # Append the row to the dataframe
            song_dict_list.append(track_dict)
    
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
    
  
def determine_next_song_index(songs_df, curr_index, already_played_indices):
    
    vars_for_sim = songs_df[ ['popularity', 'danceability',
           'energy', 'key', 'speechiness', 'acousticness', 'instrumentalness',
           'liveness', 'valence', 'tempo' ]  ] 
    
    # Standardize the variables
    scaler = StandardScaler()
    df_standardized = pd.DataFrame( scaler.fit_transform(vars_for_sim), columns = vars_for_sim.columns)
    
    #weights
    weights = {
    'popularity': 5,
    'danceability': 5,
    'energy': 6,
    'key': 5,
    'speechiness': 3,
    'acousticness': 5,
    'instrumentalness': 3,
    'liveness': 3,
    'valence': 2,
    'tempo': 20
    }
    
    for key, value in weights.items():
        df_standardized[key] = df_standardized[key] * value
    
    
    # Compute distances between observations
    distances = euclidean_distances(df_standardized)
    distances_to_curr = distances[curr_index, :]
    
    # Aritifically inflate distances for those songs that have been played
    weighter_played = np.array(songs_df['played']) * 100000000000 +1
    distances_to_curr *= weighter_played

    closest_index =  np.argsort(distances_to_curr)[1]
    
    counter = 2
    while closest_index ==curr_index or closest_index in already_played_indices:
        closest_index =  np.argsort(distances_to_curr)[counter]
        counter+=1

    return closest_index

 
def song_player(songs_df, mp3_dir):
    
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
                 while num_songs not in [str(i) for i in range(100)]:
                     num_songs = input("Enter number between 1 and 99\n")
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
                 #duration_sec = 5
                 start_time = time.time()
                 elapsed_time = 0
                 
                 print("Now playing: {} by {}...".format(songs_df['name'][curr_index] , songs_df['artists'][curr_index] ) )
                 player.play()
                 
                 
                 #Get next closest song using distance metric
                 next_song_index = determine_next_song_index(songs_df, curr_index, already_played_indices)
                 
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
                                 
                 #Set up loop to play next song
                 curr_index = next_song_index
             
             #In automatic mode, automatically set next seed and keep playing
             if automatic_mode:
                 print("*"*50, "\n\nContinuing in automatic mode - NEW SEED...\n")
                 curr_index = songs_random_indices[random_counter]
                 random_counter += 1
                 
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
             while user_input not in ['1', '2', '3']:
                 user_input = input("Try Again")
         else: 
             user_input = "1"
             
     return player
     
     
def main():
    
    
    print("\n\n", "-"*50, sep="")
    print("\n\n\nComputer DJ\n\n\n")
    print("-"*50)
    
    print("\n\nloading....")
    time.sleep(1)
    
    client_id = '2d2b3cb73d234437a56c79a7e0fcae76'
    client_secret = '8c2eb7b9be2c4b3c9f0c276bfb0db997'
    rips_archived_id = 'spotify:playlist:4Ra7m4IDwu9GVMq7FC51Kj'
    mp3_dir = '/Users/coltonlapp/Dropbox/Personal/Music/DJ/MP3s/Spotify/'
    comp_dj_dir = '/Users/coltonlapp/Dropbox/Personal/Music/DJ/SCRIPTS/ComputerDJ/'
    
    # Authenticate with Spotify API
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    all_song_ids = get_song_ids(rips_archived_id, sp)
    
    # Check if the file exists
    if not os.path.exists(comp_dj_dir + 'songs_df.csv'):
        # File does not exist, so run the build_songs function
        songs_df = build_song_df(mp3_dir, all_song_ids, sp, comp_dj_dir)
        songs_df.to_csv( comp_dj_dir + 'songs_df.csv', index=False)
    else:
        # File exists, read the CSV file into a dataframe
        songs_df = pd.read_csv(comp_dj_dir + 'songs_df.csv')
        
    print("Loaded {} songs...\n".format(len(songs_df)))
    player = song_player(songs_df, mp3_dir)
    player.stop()

if __name__ == '__main__':
    main()

   
    
    