import numpy
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from Huffman import compress, decompress
import os

from utils import chunk2bytes, bytes2chunk, idct2,dct2, maxchunkval, quantize_down, quantize_up



img = rgb2gray(imread('test.png'))
shape = img.shape

'''
Compression
'''
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

#prepare chunks for huffman encoding
ba = chunk2bytes(chunks[0],max_val)
for i in range(1,len(chunks)):
    ba += chunk2bytes(chunks[i],max_val)

#write the shape
with open('dim','w+') as f:
    f.write(str(shape[0])+','+str(shape[1])+','+str(max_val))
compress(ba,'out')



'''
DECOMPRESSION
'''

shape = [0,0]
max_val = 1
with open('dim','r') as f:
    ss = f.read()
    shape[0] = int(ss.split(',')[0])
    shape[1] = int(ss.split(',')[1])
    max_val = int(ss.split(',')[2])

bts = decompress('out')

chunks = []
for i in range(0,len(bts),64):
    c_add, good = bytes2chunk(bts[i:i+64],max_val)
    if good==False:
        break
    chunks.append(c_add)

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