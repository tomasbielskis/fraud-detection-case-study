from pymongo import MongoClient

DB_NAME = "fraud"
COLLECTION_NAME = "events"

client = MongoClient()
db = client[DB_NAME]
coll = db[COLLECTION_NAME]

# event_data = event data from data.json or the api call
# probability = model.predict()

coll.insert({'event': event_data, 'probability': probability},check_keys=False)

coll.find()
