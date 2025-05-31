import json
import argparse

def all_visited_or_not(edges,root=0):
    parent_rel={}
    all_nodes=set()
    for parent, child in edges:
        parent_rel[child]=parent
        all_nodes.add(parent)
        all_nodes.add(child)
        

parser=argparse.ArgumentParser()

parser.add_argument("--json_path",type=str,required=True,help="The json file path")

args=parser.parse_args()


with open(args.json_path,"r") as file:
    data=json.load(file)

edges=[]
for i in data['links']:
    edges.append((i['source'],i['target']))

all_visited_or_not(edges)