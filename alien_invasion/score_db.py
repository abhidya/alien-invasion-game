import pymongo
import socket
import hashlib

#connect to mongo, find the db, and find the correct collection
#update to connect to non local host
def connect_and_collect():
	client = pymongo.MongoClient("mongodb://35.188.132.96:27017")

	if "highscore_db" in client.list_database_names():
		db = client["highscore_db"]
	else:
		print("Unable to locate database")
		quit()

	if "scores" in db.list_collection_names():
		collection = db["scores"]
	else:
		print("Unable to locate collection")
		quit()

	print("Successfully connecting to db and grabbed collection")
	
	return collection

#given a name, figure out ip/name hash and update scores collection
#might have to pass the ip string in the future
def add_score(name, score, collection):
	hostname = socket.gethostname()
	IPAddr = socket.gethostbyname(hostname)
	
	hash = IPAddr+name
	hash = hashlib.md5(hash.encode()).hexdigest()
	
	query = { "key":hash }

	if collection.find(query).count() > 0:
		old_score = collection.find_one(query,{"_id":0, "score":1})["score"]
		if old_score > score:
			print("User already has a highscore")
		else:
			collection.delete_one(query)
			dict = { "key":hash, "name":name, "score":score }
			collection.insert_one(dict)
			print("Updated user's score")
	else:
		dict = { "key":hash, "name":name, "score":score }
		collection.insert_one(dict)
		print("Added new user with highscore")

#print top scores, pass -1 for all scores or n for the top n scores
#will have to be updated to communicate what to display
def print_top_scores(collection, num):
	count = 1
	docs = collection.find().sort("score")
	for doc in docs:
		print(str(count)+". "+doc["name"]+" "+str(doc["score"]))
		if (num != -1 and count == num):
			break;
		count = count + 1

#my own tests
# my_col = connect_and_collect()
# add_score("shelb", 50, my_col)
# print_top_scores(my_col, 1)
