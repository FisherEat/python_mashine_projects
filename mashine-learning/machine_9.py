"""
本案例用来展示曼德布洛特迭代模型.
使用纯粹的python 算法实现该模型, machine_10 使用tensorflow来实现该模型
绘图使用PIL
"""

import time
from PIL import Image, ImageDraw

g_size = (400, 300) # 图形最终尺寸
g_max_iteration = 256 # 最大迭代次数
g_bailout = 4 # 最大域
g_zoom = 2.5 / g_size[0] # 缩放参数
g_offset = (-g_size[0] * 0.25, 0) # 偏移量
g_HSL = (210, 80, 50) # HSL色彩基调

def draw(antialias = True):
    zi = 2 if antialias else 1 # antialias: 抗锯齿 size = [i * zi
    size = [i * zi for i in g_size]
    zoom = g_zoom / zi
    offset = [i * zi  for i in g_offset]
    bailout = g_bailout * zi
    img = Image.new("RGB", size, 0xffffff)
    dr = ImageDraw.Draw(img)

    print("painting Mandelbrot Set..")
    for xy, color in getPoints(size, offset, zoom):
        dr.point(xy, fill = color)
    print("100%n")

    del dr
    if antialias:
        img = img.resize(g_size, Image.ANTIALIAS)
    img.show()
    img.save("mandelbrot_set_%dx%d.png" % g_size)

def getPoints(size, offset, zoom, ti = 0, tstep = 1):
    "生成需要绘制的点的坐标及颜色"

    def getRepeats(c):
        z = c
        repeats = 0
        while abs(z) < g_bailout and repeats < g_max_iteration:
            z = z * z + c
            repeats += 1
        return repeats

    def getColor(r):
        color = "hsl(0, 0%, 0%)"
        if r < g_max_iteration:
            v = 1.0 * r / g_max_iteration
            h = ch * (1 - v)
            s = cs
            l = cl * (1 + v)
            color = "hsl(%d, %d%%, %d%%)" % (h, s, l)
        return color

    xs, ys = size
    xw, yh = xs / 2, ys / 2
    xo, yo = offset
    ch, cs, cl = g_HSL

    progress = 0
    for iy in range(ys):
        p = iy * 100 / ys
        if iy % 10 == 0 and p != progress:
            print ("%d%%..." % p) # 显示进度
            progress = p
        for ix in range(ti, xs, tstep):
            x = (ix - xw + xo) * zoom
            y = (iy - yh + yo) * zoom
            c = complex(x, y)
            r = getRepeats(c)
            yield (ix, iy), getColor(r)

def main():
    t0 = time.time()
    draw()
    t = time.time() - t0
    print("%dm%.3fs" % (t / 60, t % 60))

if __name__ == "__main__":
    main()