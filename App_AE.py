import numpy
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from Autoencoder import Autoencoder, Image_Dataset, encode,decode, train_autoencoder
from Huffman import compress, decompress
import os,math

from utils import add_closest_cluster, idct2,dct2, maxchunkval, quantize_down, quantize_up


model = Autoencoder()

train_autoencoder(model,Image_Dataset())


img = rgb2gray(imread('test.png'))
shape = img.shape
#seperate image into 8 * 8 chunks

chunks = []

for i in range(0,len(img), 8):
    for j in range(0, len(img[0]), 8):
        chunks.append(img[i:(i+8),j:(j+8)])

#find the max value seen in a chunk to scale them and do some DCT and quantization filtering stuff
max_val = 0
for i in range(len(chunks)):
    chunks[i] = quantize_down(dct2(chunks[i]))
    mv_t = maxchunkval(chunks[i])
    if(mv_t>max_val):
        max_val = mv_t
if max_val>128:
    max_val = 128/max_val
else:
    max_val = 1

#autoencode the chunks
encoded = []
for i in range(0,len(chunks)):
    encoded_vec = encode(chunks[i],model)
    encoded += encoded_vec

#force the encoded form onto bytes, use a really rudimentary clustering algorithm to try and get a good mapping for bytes -> floats
ba = bytearray()
bytemap = []
cluster_counts = []
for v in encoded:
    if len(bytemap) < 256:
        bytemap.append(v)
        cluster_counts.append(1)
    else:
        bytemap,cluster_counts,added_cluster = add_closest_cluster(bytemap,cluster_counts,v)
        ba+=bytearray([int(added_cluster)])

#write the shape
with open('dim','w+') as f:
    f.write(str(shape[0])+','+str(shape[1])+'\n'+str(bytemap)[1:-1])

compress(ba,'out')



'''
DECOMPRESSION
'''

shape = [0,0]
bytemap = []
with open('dim','r') as f:
    ss_l = f.read().split('\n')
    ss = ss_l[0]
    shape[0] = int(ss.split(',')[0])
    shape[1] = int(ss.split(',')[1])
    bytemap = [float(x) for x in ss_l[1].split(',')]
    

bts = decompress('out')

chunks = []
for i in range(0,len(bts),8):
    inp = numpy.ones((8))
    for j in range(8):
        if i+j>=len(bts):
            break
        inp[j] = bytemap[int(bts[i+j])]
    chunk_pred = decode(inp,model)
    chunk = numpy.ones((8,8))
    for i in range(64):
        chunk[math.floor(i/8)][i%8] = chunk_pred[i]
    chunks.append(chunk)
#reconstruct from chunks
for i in range(len(chunks)):
    chunks[i] = idct2(quantize_up(chunks[i]))

#put chunks in the right spots
img_out = numpy.ones(shape)
for i in range(0,shape[0],8):
    for j in range(0,shape[1],8):
        chunk_ind = (int(i/8) * int(shape[1]/8)) + int(j/8)
        if chunk_ind>len(chunks)-1:
            break
        img_out[i:(i+8),j:(j+8)] = chunks[chunk_ind]

print(os.path.getsize('out.aejpg') + os.path.getsize('out_tree.json'))
plt.imshow(img_out, cmap="gray")
plt.show()