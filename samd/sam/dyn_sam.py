import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm
import networkx as nx
import os
from pathlib import Path
from transformers import AutoTokenizer

class DynSAM:
   
    @dataclass
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        min_endpos: int

    def __init__(self, n_predicts: int = 40):
        self.n_predicts = n_predicts
        self.states: List[DynSAM.SAMState] = [DynSAM.SAMState(next={}, link=-1, length=0, min_endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.counter=0
        # self.tokenizer=AutoTokenizer.from_pretrained("")

        
        # params needed to be reset for each query
        self.cur_index = 0
        self.cur_length = 0
        self.currpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.graphdir = os.path.join(self.currpath, "Graphs")#Directory to store the graphs and json
        if not os.path.exists(self.graphdir):
            os.mkdir(self.graphdir)

        Path(self.graphdir,"Image").mkdir(exist_ok=True)

        self.image_path=os.path.join(self.graphdir,"Image")
        if self.counter==0:
            if os.path.exists(self.image_path):
                for items in os.listdir(self.image_path):
                    item_path=os.path.join(self.image_path,items)
                    os.remove(item_path)


    
    def reset(self):
        self.states: List[DynSAM.SAMState] = [DynSAM.SAMState(next={}, link=-1, length=0, min_endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.cur_index = 0
        self.cur_length = 0
    
    def expand_state(self, state: SAMState):
        new_index = len(self.states)
        self.states.append(state)
        return new_index

    def add_state(self, token: int):
        self.max_length += 1
        cur = self.expand_state(
            DynSAM.SAMState(
                next={}, link=-1, 
                length=self.max_length, 
                min_endpos=self.max_length
            )
        )
        p = self.last
        while p != -1 and token not in self.states[p].next:
            self.states[p].next[token] = cur
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                clone = self.expand_state(deepcopy(self.states[q]))
                self.states[clone].length = self.states[p].length + 1
                while p != -1 and self.states[p].next[token] == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur

    def transfer_state(self, index: int, length: int, token: int,is_infer=None,step=None,counter=None):
        if counter!=None:
            self.counter=counter

        while index != 0 and token not in self.states[index].next:#executed untill the condition fails
            # if is_infer==True:
            #     self.dfa_link_graph(index,step)
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:#If correct, this gets executed only one time
            # if is_infer==True:
            #     self.dfa_next_graph(index,token,step)
            index = self.states[index].next[token]
            length += 1
        else:
            index = length = 0
        if is_infer==True:
            return index, length,self.counter
        else:
            return index, length

    
    def transfer_cur_state(self, token: int):
        self.cur_index, self.cur_length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
    
    def add_tokens(self, tokens: List[int]):
        for token in tokens:
            self.transfer_cur_state(token)
            self.add_state(token)
        self.input_ids.extend(tokens)


    
    def transfer_tokens(self, tokens: List[int]):
        for token in tokens:
            self.transfer_cur_state(token)

    def lookup(self, token: int,step,counter):
        # print("The dynamic look up is invoked")
        index, length,counter = \
            self.transfer_state(self.cur_index, self.cur_length, token,is_infer=True,step=step,counter=counter)
        # print("The index and lenth returned by the dynamic look up is..",index,length)
        return index, length, counter

    def to_anc(self, index: int):
        if index != 0:
            length_to_end = self.max_length - self.states[index].min_endpos
            while self.states[index].link != 0 and self.n_predicts > length_to_end:
                index = self.states[index].link
                length_to_end = self.max_length - self.states[index].min_endpos
        return index

    def gen_draft(self, index: int, start_token: int):
        index = self.to_anc(index)
        endpos = self.states[index].min_endpos
        predicted_token=self.input_ids[endpos]
        pred_ids = [start_token] + self.input_ids[endpos + 1:endpos + self.n_predicts]
        if len(pred_ids) < self.n_predicts:
            pred_ids.extend([0] * (self.n_predicts - len(pred_ids)))
        return pred_ids
    
    #Zoho Labs Kottarakara:Visualize the dfa transitions for Dynamic-SAM

    # def dfa_next_graph(self,index,token,step):
    #     current_index=index
    #     next_index=self.states[index].next[token]
    #     self.draw((current_index,next_index),step,edge_color="black",token=token)
    # def dfa_link_graph(self,index,step):
    #     current_index=index
    #     link_index=self.states[index].link
    #     self.draw((current_index,link_index),step,edge_color="Red")
    
    # def sanitize(self,token_text):
    #     if token_text=="\n":
    #         token_text="\\\n"
    #     elif token_text == " ":
    #         token_text = "[space]"   
    #     elif token_text == "\t":
    #         token_text = "\\\t"        
    #     elif token_text.strip() == "":
    #         token_text = "[blank]"
    #     else:
    #         token_text=token_text
    #     return token_text
    
    # def draw(self,transition,step,edge_color,token=None):
    #     G=nx.DiGraph()
    #     color="lightblue"
    #     if edge_color=="black":#next condition
    #         for index in transition: 
    #             if index==0:
    #                 token_text="<ROOT>"
    #             else:
    #                 token_id=self.input_ids[self.states[index].min_endpos]
    #                 token_text=self.tokenizer.decode([token_id])
    #             token_text=str(self.sanitize(token_text))
    #             G.add_node(index,label=token_text,style="filled",fillcolor=color,fontweight="bold", fontsize="20", fontcolor="black")
    #         edge_label=self.tokenizer.decode([token])
    #         edge_label=str(self.sanitize(edge_label))
    #         G.add_edge(transition[0],transition[1],label=edge_label,color=edge_color)
    #         pydot_graph = nx.nx_pydot.to_pydot(G)
    #         pydot_graph.set_label("This graph curresponds to next in dynamic")
    #         pydot_graph.set("labelloc", "t")   # top of the graph
    #         pydot_graph.set("fontsize", "16")
    #         pydot_graph.set("fontcolor", "black")
    #         pydot_graph.set_size('"12,10!"')
    #         pydot_graph.set_ratio('fill')
    #         pydot_graph.set("dpi", "600")
    #         image_path=os.path.join(self.graphdir,"Image")
    #         image_name=f"{step}_image_{self.counter}_.png"
    #         image_absolute_path=os.path.join(image_path,image_name)
    #         pydot_graph.write_png(image_absolute_path)
    #         self.counter+=1
    #     else:#link condiition
    #         for index in transition:
    #             if index==-1:
    #                 token_text="<IMG>"
    #             elif index==0:
    #                 token_text="<ROOT>"
    #             else:
    #                 token_id=self.input_ids[self.states[index].min_endpos]
    #                 token_text=self.tokenizer.decode([token_id])
    #                 token_text=self.sanitize(token_text)
    #             G.add_node(index,label=token_text,style="filled",fillcolor=color,fontweight="bold", fontsize="20", fontcolor="black")
    #         G.add_edge(str(transition[0]),str(transition[1]),color=edge_color)
    #         pydot_graph = nx.nx_pydot.to_pydot(G)
    #         pydot_graph.set_label("The graph curresponds to link in dynamic")
    #         pydot_graph.set("labelloc", "t")   # top of the graph
    #         pydot_graph.set("fontsize", "16")
    #         pydot_graph.set("fontcolor", "black")
    #         pydot_graph.set_size('"12,10!"')
    #         pydot_graph.set_ratio('fill')
    #         pydot_graph.set("dpi", "600")
    #         image_path=os.path.join(self.graphdir,"Image")
    #         image_name=f"{step}_image_{self.counter}_.png"
    #         image_absolute_path=os.path.join(image_path,image_name)
    #         pydot_graph.write_png(image_absolute_path)          
    #         self.counter+=1
