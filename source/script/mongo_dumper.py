'''
dumpe data from mongodb:root@10.141.209.3:27017
'''
import codecs
import pymongo
from pymongo import  MongoClient


class Config(object):
    host = '10.141.209.3'
    port = 27017
    database_name = 'tweet_contents'
    table_name = 'tweets'
    fields = {'id','user:id','entities:hashtags', 'created_at'}
    data_path = "../../data/tweet_test.txt"


client = MongoClient(Config.host, Config.port)

# Get database
tweet_db = client.get_database(name = Config.database_name)

# Get table
tweet_content_tb = tweet_db.get_collection(name=Config.table_name)


def save_one_sample(user_id, text, hashtags, time):
    line = "%d\t%s\t%s\t%s\n" % (user_id, text, ";".join(hashtags), time)
    try:
        line.encode('ascii')
        writer.write(line)
    except Exception as error:
        pass


# Get items with field hashtags not empty.
# Filter is a dictionary. '$key' is a property key and 'key' is a field key
# Reference: https://docs.mongodb.com/manual/reference/
# Examples:
#     one_item = tweet_content_tb.find_one({'$where':'this.entities.hashtags.length > 0'})
#     one_item = tweet_content_tb.find_one({'entities.hashtags.1':{'$exists':True}})
# An item is a dictionary storing its information
cursor = tweet_content_tb.find({'entities.hashtags.1':{'$exists':True},'lang':'en', 'retweeted_status' : {'$exists': False}})
writer = codecs.open(Config.data_path,"w+", encoding="ascii", errors="strict")
count = 0
item_limit = 1000000
for post in cursor:
    count += 1
    if count%10000 == 0:
        print("%d posts dumped" %count)
    user_id = post.get('user').get('id')
    text = post.get('text').replace("\n"," ")
    hashtags = []
    l = post.get('entities').get('hashtags')
    for item in l:
        hashtags.append(item.get('text'))
    time = post.get('created_at')
    save_one_sample(user_id, text, hashtags, time)
    # for hashtag in hashtags:
    #     doc = text[0:text.find("#%s"%hashtag)+1]
    #     if len(doc) >= 1:
    #         line = "%d\t%s\t%s\t%s\n" %(user_id, doc, hashtag, time)
    #         try:
    #             line.encode('ascii')
    #             writer.write(line)
    #         except Exception as error:
    #             pass
    if count > item_limit:
        break
writer.close()

