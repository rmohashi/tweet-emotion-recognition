import os

from tweepy import OAuthHandler, API

class Connection:
  def __init__(self):
    self.consumer_key = os.getenv("CONSUMER_KEY")
    self.consumer_secret = os.getenv("CONSUMER_SECRET")
    self.access_token = os.getenv("ACCESS_TOKEN")
    self.access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

  def load(self):
    auth = OAuthHandler(self.consumer_key, self.consumer_secret)
    auth.set_access_token(self.access_token, self.access_token_secret)
    self.api = API(auth)
    print('Successfully connected to the Twitter API.')
