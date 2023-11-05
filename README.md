## Computer DJ
#### An automated way to transition between songs using Spotify's song metadata (tempo, energy, key, year, genre, etc...)

## Demo Link:
https://youtu.be/kguTqimvcOg

## Description:
This was a personal project I completed in the summer of 2023. My goal was to use basic ML/stats techniques to compute the similarity between songs and then automatically transition between ones that were the most similar. 
As an amateur bedroom DJ, I have a lot of mp3 files saved on my computer spanning a wide range of genres. I also have lengthy sptofiy playlists that contain thousands of songs. I wanted to have a program that mimicked the intuition of DJs for both of these mediums of song collections so that I could enjoy my disorganized collection of music more.

## Function:
This python script is able to complete two things:
1. Serve as a media player for the mp3 files on my computer, transitioning from one song to the next based on similarity scores
2. Reorganize spotify playlists to have a more optimal song transition flow

## Algorithm:
My basic insight was to create a dataset of all the songs on my computer/within a given playlist, and then compute the "similarity" between songs using ML/stats techniques. 
The way I implemented this was to rescale and weight different features, and then calculate the euclidean distance between songs. 

## Seeing the results
I collapsed the dimensionality of the songs metadata into 2 dimensions using PCA, and then showed how the song transitions occured in a scatterplot of all songs. I also included a video of the process
