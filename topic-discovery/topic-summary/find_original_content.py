import sys
import json
from util import read_df, read_json, write_json

df_path = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]

df = read_df(df_path)
#df['tweet_id'] = df['tweet_id'].astype('int64')
print ('df size : ', len(df))
df = df.set_index('tweet_id')

elems = read_json(input_path)

for elem in elems:
    elem['tweets'] = []
    for tweet_id in elem['tweet_ids']:
        row = df.loc[tweet_id]
        elem['tweets'].append({'tweet_id':tweet_id, 
                               'content':row['content'], 
                               'author_id':row['author_id']})
    elem.pop('tweet_ids')
    if 'tfidf' in elem:
        elem.pop('tfidf')
    if 'tf' in elem:
        elem.pop('tf')

write_json(elems, output_path)
