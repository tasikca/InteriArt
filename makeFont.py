#
# makeFont
#
# Builds an installable TrueType font from the central path glyphs in
# interArtAlphabet. Each stroke (a central path) is thickened into a
# filled outline with round ends, the outlines of a glyph are merged,
# and the result is written with fontTools.
#
# usage: python makeFont.py [strokeWidth] [outputFile]
#    strokeWidth - in font units, 1000 per em (default 60)
#    outputFile  - default InteriArt-Regular.ttf
#
import sys
import os
import matplotlib
matplotlib.use("Agg")   # interArtAlphabet imports pyplot, we never draw
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.agl import UV2AGL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interArtAlphabet as ia

UPM   = 1000   # font units per em
S     = 350    # font units per glyph unit, cap height (2 units) = 700
GAP   = 0.5    # space between letters in glyph units, as in drawString
CHARS = "".join(chr(i) for i in range(32, 127))   # printable ASCII

def glyphName(ch):
   # standard Adobe glyph names, e.g. 'A', 'zero', 'period'
   return UV2AGL[ord(ch)]

def toFont(x, y, strokeWidth):
   # glyph units -> font units, baseline y = -1 goes to 0 and the
   # left edge x = -1 goes to half a letter gap plus half a stroke
   return (x + 1 + GAP/2)*S + strokeWidth/2, (y + 1)*S

#
# glyphOutline
#
# Returns the merged, filled outline of one character along with
# its advance width in font units.
#
def glyphOutline(ch, strokeWidth):
   ltr,shft,scl,sz = ia.getGlyph(ch)
   shapes = []
   for s in ltr:
      for t in range(len(ltr[s]["c"])):
         cp = ia.cachedPath(ltr[s]["A"], ltr[s]["b"], ltr[s]["c"][t,:])
         x, y = toFont((cp[:,0]+shft[s][0])*scl[s][0], (cp[:,1]+shft[s][1])*scl[s][1], strokeWidth)
         # central paths have thousands of points, a 1 unit tolerance
         # is invisible at any print size
         path = LineString(np.column_stack((x, y))).simplify(1.0)
         shapes.append(path.buffer(strokeWidth/2, quad_segs=8))
   advance = int(round((sz[0] + GAP)*S + strokeWidth))
   if not shapes:
      return None, advance
   outline = unary_union(shapes).simplify(0.5)
   return outline, advance

def polygons(geom):
   if isinstance(geom, Polygon):
      return [geom]
   if isinstance(geom, MultiPolygon):
      return list(geom.geoms)
   return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]

def drawRing(pen, coords):
   pts = []
   for x, y in coords[:-1]:   # shapely repeats the first point
      p = (int(round(x)), int(round(y)))
      if not pts or p != pts[-1]:
         pts.append(p)
   if len(pts) > 1 and pts[0] == pts[-1]:
      pts.pop()
   if len(pts) < 3:
      return
   pen.moveTo(pts[0])
   for p in pts[1:]:
      pen.lineTo(p)
   pen.closePath()

def ttGlyph(outline):
   pen = TTGlyphPen(None)
   if outline is not None:
      for poly in polygons(outline):
         # TrueType wants clockwise outer contours and counter-clockwise holes
         poly = orient(poly, sign=-1.0)
         drawRing(pen, list(poly.exterior.coords))
         for hole in poly.interiors:
            if Polygon(hole).area > 4:   # drop specks where strokes meet
               drawRing(pen, list(hole.coords))
   return pen.glyph()

def notdef(advance):
   pen = TTGlyphPen(None)
   w, h, t = advance - 100, 700, 50
   for ring in ([(50,0),(50,h),(w,h),(w,0)], [(50+t,t),(w-t,t),(w-t,h-t),(50+t,h-t)]):
      pen.moveTo(ring[0])
      for p in ring[1:]:
         pen.lineTo(p)
      pen.closePath()
   return pen.glyph()

def buildFont(strokeWidth=60, fileName="InteriArt-Regular.ttf"):
   chars = CHARS
   order = [".notdef"] + [glyphName(ch) for ch in chars]
   glyphs = {".notdef": notdef(600)}
   metrics = {".notdef": (600, 50)}
   cmap = {}
   for ch in chars:
      print("Working on glyph '"+ch+"'")
      outline, advance = glyphOutline(ch, strokeWidth)
      name = glyphName(ch)
      glyphs[name] = ttGlyph(outline)
      lsb = 0 if outline is None else int(round(outline.bounds[0]))
      metrics[name] = (advance, lsb)
      cmap[ord(ch)] = name
   ascent  = int(round(2.2*S + strokeWidth))    # brackets and slashes reach y = 1.2
   descent = int(round(S + strokeWidth))        # descenders reach y = -2
   fb = FontBuilder(UPM, isTTF=True)
   fb.setupGlyphOrder(order)
   fb.setupCharacterMap(cmap)
   fb.setupGlyf(glyphs)
   fb.setupHorizontalMetrics(metrics)
   fb.setupHorizontalHeader(ascent=ascent, descent=-descent)
   fb.setupNameTable({"familyName": "InteriArt",
                      "styleName": "Regular",
                      "uniqueFontIdentifier": "InteriArt-Regular",
                      "fullName": "InteriArt Regular",
                      "psName": "InteriArt-Regular",
                      "version": "Version 1.3"})
   fb.setupOS2(sTypoAscender=ascent, sTypoDescender=-descent, sTypoLineGap=0,
               usWinAscent=ascent, usWinDescent=descent,
               sxHeight=S, sCapHeight=2*S, usWeightClass=400)
   fb.setupPost()
   fb.save(fileName)
   print("Created font file "+fileName)

if __name__ == "__main__":
   width = float(sys.argv[1]) if len(sys.argv) > 1 else 60
   name = sys.argv[2] if len(sys.argv) > 2 else "InteriArt-Regular.ttf"
   buildFont(width, name)
