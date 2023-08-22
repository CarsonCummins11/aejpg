import math,numpy
from scipy import fftpack

#run length encodes a string
def rle(s):
    d = []
    cur = ''
    for ch in s:
        if ch==cur:
            d[-1][1]+=1
        else:
            d.append([ch,1])
            cur=ch
    ret = ''
    for rl in d:
        ret+= rl[0] + str(rl[1])
    return ret
def rle_decode(s):
    ret = ''
    for i in range(0,len(s),2):
        ret+=s[i] * int(s[i+1])
    return ret

#zig zag flattens an 8*8 array
def zigzag(m):

    def get_indices(i):
        inds = [0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,4,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63]
        ind = inds[i]
        return int(math.floor(ind/8)),ind%8

    ret = ''
    for i in range(64):
        x,y = get_indices(i)
        ret+=str(m[x][y])+','
    return ret[:-1]


QUANT_MATRIX = numpy.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99],
])



#scaling the quant matrix because it makes the image appear significantly less grainy
def quantize_down(m):
    ret = numpy.ones((8,8))
    for i in range(8):
        for j in range(8):
            ret[i][j] = m[i][j]/(QUANT_MATRIX/200)[i][j]
    return ret.astype(int)

def quantize_up(m):
    ret = numpy.ones((8,8))
    for i in range(8):
        for j in range(8):
            ret[i][j] = m[i][j] * (QUANT_MATRIX)[i][j]
    return ret.astype(int)


#2D Discrete cosine transform - first dct on 0th axis, then on first to emulate 2D DCT
#automatically does type 2 for normally spaced data (because it's an image so we assume distance between pixels is consistent)
def dct2(x):
    return fftpack.dct( fftpack.dct( x, axis=0, norm='ortho' ), type=2,axis=1, norm='ortho' )

#just inverse of above
def idct2(x):
    return fftpack.idct( fftpack.idct( x, axis=0 , norm='ortho'), type=2,axis=1 , norm='ortho')


#using byte scalar to force color values to stay within 1 byte
def chunk2bytes(chunk, byte_scalar):
    flat = []
    for i in range(chunk.shape[0]):
        for j in range(chunk.shape[1]):
            flat.append(int(round(chunk[i][j] * byte_scalar))+128)
    return bytearray(flat)

def maxchunkval(chunk):
    mv = 0
    for i in range(chunk.shape[0]):
        for j in range(chunk.shape[1]):
            if abs(chunk[i][j]) > mv:
                mv = abs(chunk[i][j])
    return mv

def bytes2chunk(bts, byte_scalar):
    chunk = numpy.ones((8,8))
    if len(bts)<64:
        return chunk,False
    for i in range(64):
        chunk[math.floor(i/8),i%8] = (int(bts[i])*byte_scalar)-128
    return chunk, True

def add_closest_cluster(bytemap,cluster_counts,val):
    closest = 5000000
    c_i = 0
    for i in range(256):
        if abs(bytemap[i] - val) < closest:
            closest = abs(bytemap[i] - val)
            c_i = i
    bytemap[c_i] = ((bytemap[c_i]+(val/cluster_counts[c_i])) * cluster_counts[c_i]) / (cluster_counts[c_i]+1)
    cluster_counts[c_i]+=1
    return bytemap,cluster_counts,c_i
