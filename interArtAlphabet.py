from __future__ import division
from cpUtils import *
import numpy as np
from numpy import linalg
import scipy as sp
import matplotlib.pyplot as plt
import itertools


'''
def make_A():
   ltr = {}
   shft = {}
   scl = {}
   A,b = get2DData(4)
   c = np.array([[-1,-0.02],[1,-0.02]])
   strk = {"A":A, "b":b, "c":c}
   ltr[0] = strk
   shft[0] = [0,0]
   scl[0] = [1,1]
   sz = [2,2]
   cp = np.array([[-1,0.01],[1,-0.01]])
   strk = {"A":A, "b":b, "c":c}
   ltr[1] = strk
   shft[1] = [0,0.5]
   scl[1] = [1,2]
   return ltr,shft,scl,sz

def make_I():
   ltr = {}
   shft = {}
   scl = {}
   A,b = get2DData(4)
   c = np.array([[0.01,1],[-0.01,1],[0.01,-1],[-0.01,-1]])
   strk = {"A":A, "b":b, "c":c}
   ltr[0] = strk
   shft[0] = [0,0]
   scl[0] = [1,1]
   sz = [2,2]
   return ltr,shft,scl,sz

def make_e():
   ltr = {}
   shft = {}
   scl = {}
   # top
   A,b = get2DData(4)
   c = np.array([[-1,-0.01],[1,-0.01]])
   strk = {"A":A, "b":b, "c":c}
   ltr[0] = strk
   shft[0] = [0,0]
   scl[0] = [1,0.5]
   c = np.array([[0.01,-1]])
   strk = {"A":A, "b":b, "c":c}
   ltr[1] = strk
   shft[1] = [-1,-1]
   scl[1] = [1,0.5]
   c = np.array([[1,0]])
   strk = {"A":A, "b":b, "c":c}
   ltr[2] = strk
   shft[2] = [0,-1]
   scl[2] = [1,1]
   c = np.array([[1,0],[-1,0]])
   strk = {"A":A, "b":b, "c":c}
   ltr[3] = strk
   shft[3] = [0,-0.5]
   scl[3] = [1,1]
   sz = [2,1]
   return ltr,shft,scl,sz

def make_i():
   ltr = {}
   shft = {}
   scl = {}
   A,b = get2DData(4)
   c = np.array([[0,-1]])
   strk = {"A":A, "b":b, "c":c}
   ltr[0] = strk
   shft[0] = [-0.5,0]
   scl[0] = [1,1]
   sz = [0.5,1]
   return ltr,shft,scl,sz

def make_n():
   ltr = {}
   shft = {}
   scl = {}
   A,b = get2DData(4)
   c = np.array([[-1,-0.01],[1,-0.01]])
   strk = {"A":A, "b":b, "c":c}
   ltr[0] = strk
   shft[0] = [0,0]
   scl[0] = [1,1]
   sz = [2,1]
   return ltr,shft,scl,sz

def make_r():
   ltr = {}
   shft = {}
   scl = {}
   A,b = get2DData(4)
   c = np.array([[-1,-0.01]])
   strk = {"A":A, "b":b, "c":c}
   ltr[0] = strk
   shft[0] = [0,0]
   scl[0] = [1,1]
   sz = [0.5,1]
   return ltr,shft,scl,sz

def make_t():
   ltr = {}
   shft = {}
   scl = {}
   # top part
   A,b = get2DData(4)
   c = np.array([[-0.01,-1],[0.01,-1]])
   strk = {"A":A, "b":b, "c":c}
   ltr[0] = strk
   shft[0] = [0,0.59*0.59]
   scl[0] = [1,0.6]
   # bottom part
   c = np.array([[-0.01,1],[0.01,1]])
   strk = {"A":A, "b":b, "c":c}
   ltr[1] = strk
   shft[1] = [0,-1/0.6]
   scl[1] = [1,0.6]
   sz = [2,1.2]
   return ltr,shft,scl,sz

# --- Additional Lowercase Letters ---

def make_a():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Left bow (belly of the 'a')
   c_left = np.array([[0.01, 1], [0.01, -1]])
   ltr[0] = {"A":A, "b":b, "c":c_left}; shft[0] = [-0.5, -0.5]; scl[0] = [1, 1]
   # Stem (right side)
   c_stem = np.array([[0, 1], [0, -1]])
   ltr[1] = {"A":A, "b":b, "c":c_stem}; shft[1] = [0.5, -0.25]; scl[1] = [1, 1.25]
   sz = [1.5, 1]
   return ltr,shft,scl,sz

def make_b():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Stem (left side, tall)
   c_stem = np.array([[0, 1], [0, -1]])
   ltr[0] = {"A":A, "b":b, "c":c_stem}; shft[0] = [-0.5, 0.5]; scl[0] = [1, 1.5]
   # Right bow (belly)
   c_bow = np.array([[-0.01, 1], [-0.01, -1]])
   ltr[1] = {"A":A, "b":b, "c":c_bow}; shft[1] = [0.5, -0.5]; scl[1] = [1, 1]
   sz = [1.5, 1]
   return ltr,shft,scl,sz

def make_c():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Left-facing bow
   c_left = np.array([[0.01, 1], [0.01, -1]])
   ltr[0] = {"A":A, "b":b, "c":c_left}; shft[0] = [0, 0]; scl[0] = [1, 1]
   sz = [1.2, 1]
   return ltr,shft,scl,sz

def make_d():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Left bow (belly)
   c_bow = np.array([[0.01, 1], [0.01, -1]])
   ltr[0] = {"A":A, "b":b, "c":c_bow}; shft[0] = [-0.5, -0.5]; scl[0] = [1, 1]
   # Stem (right side, tall)
   c_stem = np.array([[0, 1], [0, -1]])
   ltr[1] = {"A":A, "b":b, "c":c_stem}; shft[1] = [0.5, 0.5]; scl[1] = [1, 1.5]
   sz = [1.5, 1]
   return ltr,shft,scl,sz

def make_h():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Stem (left side, tall)
   c_stem = np.array([[0, 1], [0, -1]])
   ltr[0] = {"A":A, "b":b, "c":c_stem}; shft[0] = [-0.5, 0.5]; scl[0] = [1, 1.5]
   # Arch (right side)
   c_arch = np.array([[-1, -0.01], [1, -0.01]])
   ltr[1] = {"A":A, "b":b, "c":c_arch}; shft[1] = [0.5, -0.5]; scl[1] = [0.8, 1]
   sz = [1.5, 1]
   return ltr,shft,scl,sz

def make_l():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Simple tall vertical stem
   c = np.array([[0, 1], [0, -1]])
   ltr[0] = {"A":A, "b":b, "c":c}; shft[0] = [0, 0]; scl[0] = [1, 1.5]
   sz = [0.5, 1]
   return ltr,shft,scl,sz

def make_m():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   c_arch = np.array([[-1, -0.01], [1, -0.01]])
   # First Arch
   ltr[0] = {"A":A, "b":b, "c":c_arch}; shft[0] = [-0.6, 0]; scl[0] = [0.8, 1]
   # Second Arch
   ltr[1] = {"A":A, "b":b, "c":c_arch}; shft[1] = [0.6, 0]; scl[1] = [0.8, 1]
   sz = [2.2, 1]
   return ltr,shft,scl,sz

def make_o():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Left bow
   c_left = np.array([[0.01, 1], [0.01, -1]])
   ltr[0] = {"A":A, "b":b, "c":c_left}; shft[0] = [-0.5, 0]; scl[0] = [1, 1]
   # Right bow
   c_right = np.array([[-0.01, 1], [-0.01, -1]])
   ltr[1] = {"A":A, "b":b, "c":c_right}; shft[1] = [0.5, 0]; scl[1] = [1, 1]
   sz = [1.8, 1]
   return ltr,shft,scl,sz

def make_s():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Top hook (bows up)
   c_top = np.array([[-1, 0.01], [1, 0.01]])
   ltr[0] = {"A":A, "b":b, "c":c_top}; shft[0] = [0, 0.5]; scl[0] = [0.8, 0.5]
   # Bottom hook (bows down)
   c_bot = np.array([[-1, -0.01], [1, -0.01]])
   ltr[1] = {"A":A, "b":b, "c":c_bot}; shft[1] = [0, -0.5]; scl[1] = [0.8, 0.5]
   # Middle diagonal connector
   c_mid = np.array([[1, -1], [-1, 1]]) 
   ltr[2] = {"A":A, "b":b, "c":c_mid}; shft[2] = [0, 0]; scl[2] = [0.6, 0.8]
   sz = [1.5, 1]
   return ltr,shft,scl,sz

def make_u():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Bottom arch (upward bias)
   c_bot = np.array([[-1, 0.01], [1, 0.01]])
   ltr[0] = {"A":A, "b":b, "c":c_bot}; shft[0] = [0, 0]; scl[0] = [1, 1]
   sz = [1.5, 1]
   return ltr,shft,scl,sz

# --- Additional Uppercase Letters ---

def make_H():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   c_stem = np.array([[0, 1], [0, -1]])
   # Left Stem
   ltr[0] = {"A":A, "b":b, "c":c_stem}; shft[0] = [-0.8, 0]; scl[0] = [1, 1.5]
   # Right Stem
   ltr[1] = {"A":A, "b":b, "c":c_stem}; shft[1] = [0.8, 0]; scl[1] = [1, 1.5]
   # Crossbar
   c_bar = np.array([[-1, 0], [1, 0]])
   ltr[2] = {"A":A, "b":b, "c":c_bar}; shft[2] = [0, 0]; scl[2] = [1.2, 1]
   sz = [2.2, 2]
   return ltr,shft,scl,sz

def make_L():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Tall Stem
   c_stem = np.array([[0, 1], [0, -1]])
   ltr[0] = {"A":A, "b":b, "c":c_stem}; shft[0] = [-0.5, 0]; scl[0] = [1, 1.5]
   # Base
   c_base = np.array([[-1, 0], [1, 0]])
   ltr[1] = {"A":A, "b":b, "c":c_base}; shft[1] = [0.5, -1.5]; scl[1] = [1.2, 1]
   sz = [2, 2]
   return ltr,shft,scl,sz

def make_O():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Large Left bow
   c_left = np.array([[0.01, 1], [0.01, -1]])
   ltr[0] = {"A":A, "b":b, "c":c_left}; shft[0] = [-0.8, 0]; scl[0] = [1, 1.5]
   # Large Right bow
   c_right = np.array([[-0.01, 1], [-0.01, -1]])
   ltr[1] = {"A":A, "b":b, "c":c_right}; shft[1] = [0.8, 0]; scl[1] = [1, 1.5]
   sz = [2.5, 2]
   return ltr,shft,scl,sz

def make_S():
   ltr, shft, scl = {}, {}, {}
   A,b = get2DData(4)
   # Top hook (bows up)
   c_top = np.array([[-1, 0.01], [1, 0.01]])
   ltr[0] = {"A":A, "b":b, "c":c_top}; shft[0] = [0, 0.8]; scl[0] = [1, 0.8]
   # Bottom hook (bows down)
   c_bot = np.array([[-1, -0.01], [1, -0.01]])
   ltr[1] = {"A":A, "b":b, "c":c_bot}; shft[1] = [0, -0.8]; scl[1] = [1, 0.8]
   # Middle diagonal connector
   c_mid = np.array([[1, -1], [-1, 1]]) 
   ltr[2] = {"A":A, "b":b, "c":c_mid}; shft[2] = [0, 0]; scl[2] = [0.8, 1.2]
   sz = [2.2, 2]
   return ltr,shft,scl,sz
'''

#
# Stroke primitives
#
# Every stroke is a central path in the square {x : Ax <= b} from
# get2DData(4). Such a path always starts at the center of the square
# and ends at either the middle of a facet or a vertex:
#
#   c = d           straight stroke from the center to a facet
#   c = d1 + EPS*d2 rounded elbow, out along d1 and then turning to d2
#   c = d1 + SOFT*d2 soft curve from the center to a vertex
#
# These are exactly the strokes of the hand-made I, n, e, r, and t
# above. Placing a stroke means choosing where the center of the square
# goes (start) and how far each arm reaches (size). drawString maps a
# path point p to (p + shft)*scl, so shft = start/size and scl = size.
#
# Glyph coordinates: baseline y = -1, x-height y = 0, cap/ascender
# height y = 1, descender y = -2, and glyphs start at x = -1.
#
SQ_A, SQ_b = get2DData(4)
EPS  = 0.01   # elbow corners, matches the hand-made letters
SOFT = 0.35   # curved diagonals (K, N, X, ...)
VEE  = 0.6    # straighter curves for V-shapes (V, W, v, w, Y)

def place(c, start, size):
   size = np.array(size, dtype=float)
   size[size == 0] = 1   # unused axis of a straight stroke
   start = np.array(start, dtype=float)
   strk = {"A":SQ_A, "b":SQ_b, "c":np.array([c], dtype=float)}
   return strk, list(start/size), list(size)

def line(start, end):
   # straight stroke, any slope works since scaling keeps it straight
   delta = np.subtract(end, start)
   return [place(np.sign(delta), start, np.abs(delta))]

def elbow(start, corner, end, eps=EPS):
   d1 = np.sign(np.subtract(corner, start))
   d2 = np.sign(np.subtract(end, corner))
   size = np.abs(np.subtract(corner, start)) + np.abs(np.subtract(end, corner))
   return [place(d1 + eps*d2, start, size)]

def soft(start, end, lead="h", eps=SOFT):
   # leaves start horizontally (lead="h") or vertically (lead="v")
   # and bends into end
   delta = np.subtract(end, start)
   d = np.sign(delta)
   c = [d[0], eps*d[1]] if lead == "h" else [eps*d[0], d[1]]
   return [place(c, start, np.abs(delta))]

def poly(*pts):
   # rectilinear polyline with rounded corners, one elbow per corner,
   # neighbouring elbows meet at the midpoints of the segments
   pts = [np.array(p, dtype=float) for p in pts]
   n = len(pts) - 1
   if n == 1:
      return line(pts[0], pts[1])
   strokes = []
   for k in range(1, n):
      a = pts[0] if k == 1 else 0.5*(pts[k-1] + pts[k])
      e = pts[n] if k == n-1 else 0.5*(pts[k] + pts[k+1])
      strokes += elbow(a, pts[k], e)
   return strokes

def glyph(width, *strokeLists):
   ltr, shft, scl = {}, {}, {}
   for s, (strk, sh, sc) in enumerate(itertools.chain(*strokeLists)):
      ltr[s], shft[s], scl[s] = strk, sh, sc
   return ltr, shft, scl, [width, 2]

# --- UPPERCASE ALPHABET ---

def make_A():
   # original hand-made A
   ltr = {}
   shft = {}
   scl = {}
   A,b = get2DData(4)
   c = np.array([[-1,-0.02],[1,-0.02]])
   ltr[0] = {"A":A, "b":b, "c":c}
   shft[0] = [0,0]
   scl[0] = [1,1]
   ltr[1] = {"A":A, "b":b, "c":c}
   shft[1] = [0,0.5]
   scl[1] = [1,2]
   return ltr,shft,scl,[2,2]

def make_B():
   return glyph(2, line((-1,1),(-1,-1)),
                   poly((-1,1),(0.6,1),(0.6,0),(-1,0)),
                   poly((-1,0),(1,0),(1,-1),(-1,-1)))

def make_C():
   return glyph(2, poly((1,0.5),(1,1),(-1,1),(-1,-1),(1,-1),(1,-0.5)))

def make_D():
   return glyph(2, line((-1,1),(-1,-1)),
                   poly((-1,1),(1,1),(1,-1),(-1,-1)))

def make_E():
   return glyph(2, poly((1,1),(-1,1),(-1,-1),(1,-1)),
                   line((-1,0),(0.6,0)))

def make_F():
   return glyph(2, poly((1,1),(-1,1),(-1,-1)),
                   line((-1,0),(0.6,0)))

def make_G():
   return glyph(2, poly((1,0.5),(1,1),(-1,1),(-1,-1),(1,-1),(1,0),(0,0)))

def make_H():
   # the hand-made I turned on its side
   return glyph(2, elbow((0,0),(-1,0),(-1,1)), elbow((0,0),(-1,0),(-1,-1)),
                   elbow((0,0),(1,0),(1,1)),   elbow((0,0),(1,0),(1,-1)))

def make_I():
   # original hand-made I
   ltr = {}
   shft = {}
   scl = {}
   A,b = get2DData(4)
   c = np.array([[0.01,1],[-0.01,1],[0.01,-1],[-0.01,-1]])
   ltr[0] = {"A":A, "b":b, "c":c}
   shft[0] = [0,0]
   scl[0] = [1,1]
   return ltr,shft,scl,[2,2]

def make_J():
   return glyph(2, elbow((0.4,0),(0.4,1),(1,1)), elbow((0.4,0),(0.4,1),(-0.2,1)),
                   poly((0.4,0),(0.4,-1),(-1,-1),(-1,-0.4)))

def make_K():
   return glyph(2, line((-1,1),(-1,-1)),
                   soft((-1,0),(1,1)), soft((-1,0),(1,-1)))

def make_L():
   return glyph(2, poly((-1,1),(-1,-1),(1,-1)))

def make_M():
   return glyph(2, line((-1,1),(-1,-1)), line((1,1),(1,-1)),
                   soft((0,-0.2),(-1,1),"v"), soft((0,-0.2),(1,1),"v"))

def make_N():
   return glyph(2, line((-1,1),(-1,-1)), line((1,1),(1,-1)),
                   soft((-1,1),(1,-1),"v"))

def make_O():
   return glyph(2, poly((0,1),(1,1),(1,-1),(-1,-1),(-1,1),(0,1)))

def make_P():
   return glyph(2, line((-1,1),(-1,-1)),
                   poly((-1,1),(1,1),(1,0),(-1,0)))

def make_Q():
   return glyph(2, poly((0,1),(1,1),(1,-1),(-1,-1),(-1,1),(0,1)),
                   soft((0.2,-0.4),(1.2,-1.4),"v"))

def make_R():
   return glyph(2, line((-1,1),(-1,-1)),
                   poly((-1,1),(1,1),(1,0),(-1,0)),
                   soft((-0.2,0),(1,-1),"h"))

def make_S():
   return glyph(2, poly((1,0.5),(1,1),(-1,1),(-1,0),(1,0),(1,-1),(-1,-1),(-1,-0.5)))

def make_T():
   return glyph(2, elbow((0,0),(0,1),(-1,1)), elbow((0,0),(0,1),(1,1)),
                   line((0,0),(0,-1)))

def make_U():
   # the hand-made n turned upside down
   return glyph(2, elbow((0,-1),(-1,-1),(-1,1)), elbow((0,-1),(1,-1),(1,1)))

def make_V():
   return glyph(2, soft((0,-1),(-1,1),"h",VEE), soft((0,-1),(1,1),"h",VEE))

def make_W():
   return glyph(3, soft((-0.25,-1),(-1,1),"h",VEE), soft((-0.25,-1),(0.5,0.4),"h",VEE),
                   soft((1.25,-1),(0.5,0.4),"h",VEE), soft((1.25,-1),(2,1),"h",VEE))

def make_X():
   return glyph(2, soft((0,0),(-1,1)), soft((0,0),(1,1)),
                   soft((0,0),(-1,-1)), soft((0,0),(1,-1)))

def make_Y():
   return glyph(2, soft((0,0),(-1,1),"h",VEE), soft((0,0),(1,1),"h",VEE),
                   line((0,0),(0,-1)))

def make_Z():
   return glyph(2, line((-1,1),(1,1)), soft((1,1),(-1,-1),"v"),
                   line((-1,-1),(1,-1)), line((-0.6,0),(0.6,0)))

# --- LOWERCASE ALPHABET ---
#
# Most lowercase letters sit in a 1 x 1 square, x in [-1,0] and
# y in [-1,0], with ascenders up to 1 and descenders down to -2.
#
LL, LM, LR = -1, -0.5, 0   # left, middle, and right of the square

def make_a():
   return glyph(1, elbow((LM,0),(LL,0),(LL,-0.25)), elbow((LM,0),(LR,0),(LR,-1)),
                   poly((LR,-0.5),(LL,-0.5),(LL,-1),(LR,-1)))

def make_b():
   return glyph(1, line((LL,1),(LL,-1)),
                   poly((LL,0),(LR,0),(LR,-1),(LL,-1)))

def make_c():
   return glyph(1, poly((LR,-0.25),(LR,0),(LL,0),(LL,-1),(LR,-1),(LR,-0.75)))

def make_d():
   return glyph(1, line((LR,1),(LR,-1)),
                   poly((LR,0),(LL,0),(LL,-1),(LR,-1)))

def make_e():
   # the hand-made e, same strokes at half the width
   return glyph(1, elbow((LM,0),(LL,0),(LL,-0.5)), elbow((LM,0),(LR,0),(LR,-0.5)),
                   line((LM,-0.5),(LL,-0.5)), line((LM,-0.5),(LR,-0.5)),
                   elbow((LL,-0.5),(LL,-1),(LM,-1)), line((LM,-1),(LR,-1)))

def make_f():
   return glyph(1, poly((LR,0.7),(LR,1),(-0.6,1),(-0.6,-1)),
                   line((LL,0),(LR,0)))

def make_g():
   return glyph(1, poly((LR,0),(LL,0),(LL,-1),(LR,-1)),
                   poly((LR,0),(LR,-2),(LL,-2),(LL,-1.7)))

def make_h():
   return glyph(1, line((LL,1),(LL,-1)), elbow((LL,0),(LR,0),(LR,-1)))

def make_i():
   return glyph(0, line((LL,0),(LL,-1)), line((LL,0.35),(LL,0.6)))

def make_j():
   return glyph(1, poly((LR,0),(LR,-2),(LL,-2),(LL,-1.7)), line((LR,0.35),(LR,0.6)))

def make_k():
   return glyph(1, line((LL,1),(LL,-1)),
                   soft((LL,-0.5),(LR,0)), soft((LL,-0.5),(LR,-1)))

def make_l():
   return glyph(0, line((LL,1),(LL,-1)))

def make_m():
   # two n-arches sharing the middle leg
   return glyph(1.5, elbow((-0.625,0),(-1,0),(-1,-1)), elbow((-0.625,0),(-0.25,0),(-0.25,-1)),
                     elbow((0.125,0),(-0.25,0),(-0.25,-1)), elbow((0.125,0),(0.5,0),(0.5,-1)))

def make_n():
   # the hand-made n at half the width
   return glyph(1, elbow((LM,0),(LL,0),(LL,-1)), elbow((LM,0),(LR,0),(LR,-1)))

def make_o():
   return glyph(1, poly((LM,0),(LR,0),(LR,-1),(LL,-1),(LL,0),(LM,0)))

def make_p():
   return glyph(1, line((LL,0),(LL,-2)),
                   poly((LL,0),(LR,0),(LR,-1),(LL,-1)))

def make_q():
   return glyph(1, line((LR,0),(LR,-2)),
                   poly((LR,0),(LL,0),(LL,-1),(LR,-1)))

def make_r():
   # original hand-made r, it already fills the square
   ltr = {}
   shft = {}
   scl = {}
   A,b = get2DData(4)
   c = np.array([[-1,-0.01]])
   ltr[0] = {"A":A, "b":b, "c":c}
   shft[0] = [0,0]
   scl[0] = [1,1]
   return ltr,shft,scl,[1,1]

def make_s():
   return glyph(1, poly((LR,-0.2),(LR,0),(LL,0),(LL,-0.5),(LR,-0.5),(LR,-1),(LL,-1),(LL,-0.8)))

def make_t():
   # the hand-made plus-shaped t, stretched up to the ascender line;
   # the crossbar comes from the flares meeting at the x-height
   return glyph(1, elbow((LM,1),(LM,0),(LL,0)),  elbow((LM,1),(LM,0),(LR,0)),
                   elbow((LM,-1),(LM,0),(LL,0)), elbow((LM,-1),(LM,0),(LR,0)))

def make_u():
   return glyph(1, elbow((LM,-1),(LL,-1),(LL,0)), elbow((LM,-1),(LR,-1),(LR,0)))

def make_v():
   # V at x-height
   return glyph(1, soft((LM,-1),(LL,0),"h",VEE), soft((LM,-1),(LR,0),"h",VEE))

def make_w():
   return glyph(1.5, soft((-0.625,-1),(-1,0),"h",VEE), soft((-0.625,-1),(-0.25,-0.3),"h",VEE),
                     soft((0.125,-1),(-0.25,-0.3),"h",VEE), soft((0.125,-1),(0.5,0),"h",VEE))

def make_x():
   return glyph(1, soft((LM,-0.5),(LL,0)), soft((LM,-0.5),(LR,0)),
                   soft((LM,-0.5),(LL,-1)), soft((LM,-0.5),(LR,-1)))

def make_y():
   return glyph(1, poly((LL,0),(LL,-1),(LR,-1)),
                   poly((LR,0),(LR,-2),(LL,-2),(LL,-1.7)))

def make_z():
   return glyph(1, line((LL,0),(LR,0)), soft((LR,0),(LL,-1),"v"),
                   line((LL,-1),(LR,-1)), line((-0.8,-0.5),(-0.2,-0.5)))

# --- NUMBERS ---

def make_0():
   # narrower than O, with a center tick to tell them apart
   return glyph(1.6, poly((-0.2,1),(0.6,1),(0.6,-1),(-1,-1),(-1,1),(-0.2,1)),
                     line((-0.2,0.3),(-0.2,-0.3)))

def make_1():
   return glyph(2, poly((0,0),(0,1),(-0.7,1),(-0.7,0.6)),
                   elbow((0,0),(0,-1),(-1,-1)), elbow((0,0),(0,-1),(1,-1)))

def make_2():
   return glyph(2, poly((-1,0.5),(-1,1),(1,1),(1,0),(-1,0),(-1,-1),(1,-1)))

def make_3():
   return glyph(2, poly((-1,0.5),(-1,1),(1,1),(1,-1),(-1,-1),(-1,-0.5)),
                   line((1,0),(-0.4,0)))

def make_4():
   return glyph(2, poly((-1,1),(-1,-0.3),(1,-0.3)), line((0.4,0.5),(0.4,-1)))

def make_5():
   return glyph(2, poly((1,1),(-1,1),(-1,0),(1,0),(1,-1),(-1,-1),(-1,-0.5)))

def make_6():
   return glyph(2, poly((1,0.5),(1,1),(-1,1),(-1,-1),(1,-1),(1,0),(-1,0)))

def make_7():
   return glyph(2, poly((-1,0.5),(-1,1),(1,1)), soft((1,1),(-0.2,-1),"v",VEE))

def make_8():
   return glyph(2, poly((0,1),(0.8,1),(0.8,0),(-0.8,0),(-0.8,1),(0,1)),
                   poly((0,0),(1,0),(1,-1),(-1,-1),(-1,0),(0,0)))

def make_9():
   return glyph(2, poly((-1,-0.5),(-1,-1),(1,-1),(1,1),(-1,1),(-1,0),(1,0)))

# --- PUNCTUATION AND SYMBOLS ---
#
# Dots are short strokes like the dot on i. Symbols that sit with
# lowercase text (+ = < > ~) are centered on y = -0.25.
#
DOT = 0.25   # length of a dot

def make_period():
   return glyph(0, line((-1,-1),(-1,-1+DOT)))

def make_comma():
   return glyph(0.25, soft((-0.75,-1+DOT),(-1,-1.3),"v"))

def make_colon():
   return glyph(0, line((-1,-DOT),(-1,0)), line((-1,-1),(-1,-1+DOT)))

def make_semicolon():
   return glyph(0.25, line((-0.75,-DOT),(-0.75,0)), soft((-0.75,-1+DOT),(-1,-1.3),"v"))

def make_exclam():
   return glyph(0, line((-1,1),(-1,-0.5)), line((-1,-1),(-1,-1+DOT)))

def make_question():
   return glyph(1.4, poly((-1,0.5),(-1,1),(0.4,1),(0.4,0),(-0.3,0),(-0.3,-0.5)),
                     line((-0.3,-1),(-0.3,-1+DOT)))

def make_quotesingle():
   return glyph(0, line((-1,1),(-1,0.5)))

def make_quotedbl():
   return glyph(0.4, line((-1,1),(-1,0.5)), line((-0.6,1),(-0.6,0.5)))

def make_grave():
   return glyph(0.4, soft((-1,1),(-0.6,0.7),"v"))

def make_hyphen():
   return glyph(1, line((-1,-0.25),(0,-0.25)))

def make_underscore():
   return glyph(2, line((-1,-1.3),(1,-1.3)))

def make_plus():
   # the plus-shaped t at symbol size
   return glyph(1.2, elbow((-0.4,0.35),(-0.4,-0.25),(-1,-0.25)),  elbow((-0.4,0.35),(-0.4,-0.25),(0.2,-0.25)),
                     elbow((-0.4,-0.85),(-0.4,-0.25),(-1,-0.25)), elbow((-0.4,-0.85),(-0.4,-0.25),(0.2,-0.25)))

def make_equal():
   return glyph(1.2, line((-1,0),(0.2,0)), line((-1,-0.5),(0.2,-0.5)))

def make_less():
   return glyph(1.2, soft((-1,-0.25),(0.2,0.35)), soft((-1,-0.25),(0.2,-0.85)))

def make_greater():
   return glyph(1.2, soft((0.2,-0.25),(-1,0.35)), soft((0.2,-0.25),(-1,-0.85)))

def make_asciitilde():
   # a rounded meander
   return glyph(1.2, poly((-1,-0.45),(-1,-0.05),(-0.4,-0.05),(-0.4,-0.45),(0.2,-0.45),(0.2,-0.05)))

def make_asciicircum():
   # V turned upside down
   return glyph(1, soft((-0.5,1),(-1,0.3),"h",VEE), soft((-0.5,1),(0,0.3),"h",VEE))

def make_asterisk():
   return glyph(1, soft((-0.5,0.55),(-1,0.9)), soft((-0.5,0.55),(0,0.9)),
                   soft((-0.5,0.55),(-1,0.2)), soft((-0.5,0.55),(0,0.2)),
                   line((-0.5,1),(-0.5,0.1)))

def make_slash():
   return glyph(1.2, line((-1,-1.2),(0.2,1.2)))

def make_backslash():
   return glyph(1.2, line((-1,1.2),(0.2,-1.2)))

def make_bar():
   return glyph(0, line((-1,1.2),(-1,-1.6)))

def make_parenleft():
   return glyph(0.7, soft((-1,-0.1),(-0.3,1.1),"v"), soft((-1,-0.1),(-0.3,-1.3),"v"))

def make_parenright():
   return glyph(0.7, soft((-0.3,-0.1),(-1,1.1),"v"), soft((-0.3,-0.1),(-1,-1.3),"v"))

def make_bracketleft():
   return glyph(0.7, poly((-0.3,1.1),(-1,1.1),(-1,-1.3),(-0.3,-1.3)))

def make_bracketright():
   return glyph(0.7, poly((-1,1.1),(-0.3,1.1),(-0.3,-1.3),(-1,-1.3)))

def make_braceleft():
   return glyph(0.8, poly((-1,-0.1),(-0.6,-0.1),(-0.6,1.1),(-0.2,1.1)),
                     poly((-1,-0.1),(-0.6,-0.1),(-0.6,-1.3),(-0.2,-1.3)))

def make_braceright():
   return glyph(0.8, poly((-0.2,-0.1),(-0.6,-0.1),(-0.6,1.1),(-1,1.1)),
                     poly((-0.2,-0.1),(-0.6,-0.1),(-0.6,-1.3),(-1,-1.3)))

def make_at():
   # a lowercase a inside a hooked frame
   return glyph(2, poly((0.2,0.3),(-0.5,0.3),(-0.5,-0.5),(0.2,-0.5)), line((0.2,0.3),(0.2,-0.5)),
                   poly((0.2,-0.5),(1,-0.5),(1,1),(-1,1),(-1,-1),(1,-1)))

def make_numbersign():
   return glyph(1.6, line((-0.5,0.7),(-0.5,-1)), line((0.1,0.7),(0.1,-1)),
                     line((-1,0.2),(0.6,0.2)), line((-1,-0.5),(0.6,-0.5)))

def make_dollar():
   return glyph(1.6, poly((0.6,0.5),(0.6,0.9),(-1,0.9),(-1,0),(0.6,0),(0.6,-0.9),(-1,-0.9),(-1,-0.5)),
                     line((-0.2,1.2),(-0.2,-1.2)))

def make_percent():
   return glyph(1.8, poly((-0.7,1),(-0.4,1),(-0.4,0.4),(-1,0.4),(-1,1),(-0.7,1)),
                     poly((0.5,-0.4),(0.8,-0.4),(0.8,-1),(0.2,-1),(0.2,-0.4),(0.5,-0.4)),
                     line((-0.9,-1),(0.7,1)))

def make_ampersand():
   # an 8 whose lower right corner kicks out
   return glyph(1.6, poly((-0.4,1),(0,1),(0,0.2),(-0.8,0.2),(-0.8,1),(-0.4,1)),
                     poly((-0.4,0.2),(-1,0.2),(-1,-1),(0.2,-1),(0.2,-0.2),(0.6,-0.2)))

# characters that can't be part of a function name
PUNCT = {'.':make_period, ',':make_comma, ':':make_colon, ';':make_semicolon,
         '!':make_exclam, '?':make_question, "'":make_quotesingle, '"':make_quotedbl,
         '`':make_grave, '-':make_hyphen, '_':make_underscore, '+':make_plus,
         '=':make_equal, '<':make_less, '>':make_greater, '~':make_asciitilde,
         '^':make_asciicircum, '*':make_asterisk, '/':make_slash, '\\':make_backslash,
         '|':make_bar, '(':make_parenleft, ')':make_parenright, '[':make_bracketleft,
         ']':make_bracketright, '{':make_braceleft, '}':make_braceright, '@':make_at,
         '#':make_numbersign, '$':make_dollar, '%':make_percent, '&':make_ampersand}

def make_space():
   # narrower than a letter so words don't drift apart
   return {}, {}, {}, [0.4, 1]

#
# getGlyph
#
# Returns the stroke data for one character.
#
def getGlyph(ch):
   if ch == ' ':
      return make_space()
   if ch in PUNCT:
      return PUNCT[ch]()
   fn = globals().get('make_'+ch)
   if fn is None:
      print("No glyph for '"+ch+"', leaving a space")
      return make_space()
   return fn()

#
# cachedPath
#
# Glyph strokes reuse a handful of c-vectors in the same square, so we
# only compute each central path once. Paths come from the Newton
# placement technique with the same settings plot2DFlower uses.
#
MAXDIST = 0.001   # maximum distance error to the path
THETA   = 0.5     # convex combo used to place new points

_cpCache = {}
def cachedPath(A,b,c):
   key = (A.tobytes(), tuple(np.round(np.ravel(c),10)))
   if key not in _cpCache:
      _cpCache[key] = generateNewtonPath(A,b,c,MAXDIST,THETA)
   return _cpCache[key]

def drawString(str, *args):
   # See if we want to translate, rotate, or color
   # NOTE: should add scaling as well
   if len(args) == 0:
      theta = 0
      T = [0,0]
      colr = "k"
   elif len(args) == 1:
      theta = args[0]
      T = [0,0]
      colr = "k"
   elif len(args) == 2:
      theta = args[0]
      T = list(args[1])
      colr = "k"
   elif len(args) == 3:
      theta = args[0]
      T = list(args[1])
      colr = args[2]
   else:
      print("Error: Unknown number of optional arguments in drawString")
   #
   # Plot the letters
   #
   #fig = plt.figure()
   fig = plt.figure(figsize=(2*len(str),4))
   for chr in str:
      ltr,shft,scl,sz = getGlyph(chr)
      for s in ltr:
         for t in range(len(ltr[s]["c"])):
            cp = cachedPath(ltr[s]["A"],ltr[s]["b"],ltr[s]["c"][t,:])
            # place the stroke, then translate (T is not scaled)
            x = (cp[:,0]+shft[s][0])*scl[s][0] + T[0]
            y = (cp[:,1]+shft[s][1])*scl[s][1] + T[1]
            plt.plot(np.cos(theta)*x + np.sin(theta)*y, \
                    -np.sin(theta)*x + np.cos(theta)*y, \
                     linewidth=5, color=colr)
      T[0]=T[0]+sz[0]+0.5
   plt.axis("off")
   plt.gca().set_aspect('equal')
   #plt.savefig("InteriArt.png")
   #plt.savefig("InteriArtTrBkg.png", transparent=True)
   plt.show()

if __name__ == "__main__":
   drawString('Interi',0,[0,0],'#3C0008')
