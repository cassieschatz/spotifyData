#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:04:37 2021

@author: cassieschatz
"""
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import pandas as pd

#Initial access:
'''
credentials = json.load(open('authorization.json'))
client_id = credentials['client_id']
client_secret = credentials['client_secret']

playlist_index = 0

playlists = json.load(open('playlists.json'))
playlist_uri = playlists[playlist_index]['uri']
'''

client_id = 'f74562884bc54662b03146c144c5c411'
client_secret = '9119de29553e47cdaeae7427379c0ec7'

#https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=52327ab78f564998
playlist_uri = 'spotify:user:littlemisshoran:playlist:37i9dQZF1DXcBWIGoYBM5M'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id,client_secret=client_secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

uri = playlist_uri    # the URI is split by ':' to get the username and playlist ID
username = uri.split(':')[2]
playlist_id = uri.split(':')[4]

results = sp.user_playlist(username, playlist_id, 'tracks')


#Looping through and getting all of the songs:
playlist_tracks_data = results['tracks']
playlist_tracks_id = []
playlist_tracks_titles = []
playlist_tracks_artists = []
playlist_tracks_first_artists = []
playlist_pop = []

for track in playlist_tracks_data['items']:
    playlist_tracks_id.append(track['track']['id'])
    playlist_tracks_titles.append(track['track']['name'])
    # adds a list of all artists involved in the song to the list of artists for the playlist
    artist_list = []
    for artist in track['track']['artists']:
        artist_list.append(artist['name'])
    playlist_tracks_artists.append(artist_list)
    playlist_tracks_first_artists.append(artist_list[0])
    #playlist_pop = sp.track(track)



for track in playlist_tracks_data['items']:
    playlist_pop.append(track['track']['popularity'])

features = sp.audio_features(playlist_tracks_id)


features_df = pd.DataFrame(data=features, columns=features[0].keys())
features_df['title'] = playlist_tracks_titles
features_df['first_artist'] = playlist_tracks_first_artists
features_df['all_artists'] = playlist_tracks_artists
features_df['popularity'] = playlist_pop
#features_df['popularity'] = playlist_pop
#features_df = features_df.set_index('id')
features_df = features_df[['id', 'title', 'first_artist', 'all_artists',
                           'danceability', 'energy', 'key', 'loudness',
                           'mode', 'acousticness', 'instrumentalness',
                           'liveness', 'valence', 'tempo',
                           'duration_ms', 'time_signature','popularity']]
features_df.to_csv('rawData.csv', mode = 'a', header=False)

