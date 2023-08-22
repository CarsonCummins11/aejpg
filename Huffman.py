import json
import bitstring

import numpy as np
class Node:
    def __init__(self, occurences, left=None,right=None,value=''):
        self.value = value
        self.occurences = occurences
        self.right=right
        self.left=left
    def get_symbol_dict(self,d,path):
        if not self.value=='':
            d[self.value] = path
            return d
        if(not self.left==None):
            d = self.left.get_symbol_dict(d,path+'0')
        if(not self.right==None):
           d = self.right.get_symbol_dict(d,path+'1')
        return d
    def get_symbol(self,full_path):
        if len(full_path)==0:
            return None
        if self.value=='':
            if(full_path[0]=='1'):
                return self.right.get_symbol(full_path[1:])
            else:
                return self.left.get_symbol(full_path[1:])
        else:
            return self.value,full_path

def deserialize_tree(t):
    occurences = json.loads(t)
    forest = []
    for k,v in occurences.items():
        forest.append(
            Node(v,value=k)
        )
    while(len(forest)>1):
        forest = sorted(forest, key=lambda node: node.occurences)
        n1 = forest.pop(0)
        n2 = forest.pop(0)
        n_new = Node(n1.occurences+n2.occurences,left=n1,right=n2)
        forest.append(n_new)
    return forest[0]
    

def decompress(from_file):
    bin_info = from_file+'.aejpg'
    tree_info = from_file+'_tree.json'

    tree = deserialize_tree(open(tree_info).read())
    inf = str(bitstring.BitArray(open(bin_info,'rb').read()).bin)
    ret = bytearray()
    sym = tree.get_symbol(inf)
    while(not sym == None):
        ret+=bytearray([int(sym[0])])
        inf = sym[1]
        sym  = tree.get_symbol(inf)
    return ret

def compress(inf,to_file):
    occurences = {}
    for item in inf:
        if item in occurences:
            occurences[item]+=1
        else:
            occurences[item]=1
    forest = []
    for k,v in occurences.items():
        forest.append(
            Node(v,value=k)
        )
    while(len(forest)>1):
        forest = sorted(forest, key=lambda node: node.occurences)
        n1 = forest.pop(0)
        n2 = forest.pop(0)
        n_new = Node(n1.occurences+n2.occurences,left=n1,right=n2)
        forest.append(n_new)
    ret = ''

    symbols = forest[0].get_symbol_dict({},'')

    for sym in inf:
        ret+=symbols[sym]
    ret = bitstring.BitArray(bin=ret).tobytes()


    open(to_file+'.aejpg','wb').write(ret)
    open(to_file+'_tree.json','w+').write(json.dumps(occurences))