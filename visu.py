# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:16:20 2026

@author: ARJASK
"""

import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Data_cropper import get_img_np, crop_rectangle_data

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.constants import c, hbar
from scipy.constants import e as eV
scale = hbar*c/eV
def lam2eV(lam):
    return hbar*2*np.pi*c/lam/eV

def R(th):
    return np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])

def calc_mn(m,n):
    return m*m + n*n + m*n


def get_idx(kmax):

    N = int(np.ceil(2*kmax))

    pp = []
    for i in range(-N,N+1):
        for j in range(-N,N+1):
            if calc_mn(i,j) > (kmax*2)**2:
                continue
            pp.append([i,j])
    return pp


class circ:
    def __init__(self, x,y,r, DO = [], Round = 6):
        self.x = np.round(x,Round)
        self.y = np.round(y,Round)
        self.r = np.round(r,Round)
        self.DO = DO

    def __eq__(self, other):
        c1 = np.isclose(self.x, other.x)
        c2 = np.isclose(self.y, other.y)
        c3 = np.isclose(self.r, other.r)
        return c1 and c2 and c3

    def __add__(self, other):
        return circ(self.x, self.y, self.r, self.DO + other.DO)
    @property
    def S(self):
        return 1#sum([S(i,j) for i,j in self.DO])


def fit_circle(pp, plot = False):
    x1,y1 = pp[0]
    x2,y2 = pp[1]
    x3,y3 = pp[2]
    A = np.array([
        [2*x1, 2*y1, 1],
        [2*x2, 2*y2, 1],
        [2*x3, 2*y3, 1]
    ])
    b = np.array([
        [-(x1*x1 + y1*y1)],
        [-(x2*x2 + y2*y2)],
        [-(x3*x3 + y3*y3)],
    ])
    if np.linalg.det(A) == 0:
        return 0,0,np.inf
    a,b,c = np.linalg.solve(A,b)

    xc = np.round(-a,8)
    yc = np.round(-b,8)
    r = np.round(np.sqrt(xc*xc + yc*yc - c),6)

    if plot:
        fig, ax = plt.subplots(1,figsize = (4,4))
        ax.plot([x1,x2,x3],[y1,y2,y3],'o')
        T = np.linspace(0,2*np.pi,100)
        X = np.cos(T)*r + xc
        Y = np.sin(T)*r + yc
        ax.plot(X,Y,'--')
        ax.plot(xc,yc, 'o')

    return xc[0],yc[0],r[0]

def gen_labels(runiq):
    runiq = np.round(runiq**2, 3)
    
    Gammas = runiq == runiq.astype(np.int32)
    
    r4 = np.round(4*runiq, 3)
    M = (r4 == r4.astype(np.int32)) & ~Gammas

    r3 = np.round(3*runiq,2)
    K = (r3 == r3.astype(np.int32)) & ~Gammas

    g_idx = 1
    m_idx = 1
    t_idx = 1
    k_idx = 1
    labs = []
    for i in range(len(runiq)):
        if Gammas[i]:
            labs.append("G{}".format(g_idx))
#            labs["G{}".format(g_idx)] = i
            g_idx += 1
        elif M[i]:
            labs.append("M{}".format(m_idx))
#            labs["M{}".format(m_idx)] = i
            m_idx += 1
        elif K[i]:
            labs.append("K{}".format(k_idx))
#            labs["K{}".format(k_idx)] = i
            k_idx += 1
        else:
            labs.append("T{}".format(t_idx))
#            labs["T{}".format(t_idx)] = i
            t_idx += 1
            
    return labs


def calc_HSPs(points, b1, b2, k):

    combs = combinations(points,3)
    pairs = combinations(points,2)
    
    circs = []
    for c_ in combs:
        pp = np.array([i*b1 + j*b2 for i,j in c_])
        
        x,y,r = fit_circle(pp)
        c_ = circ(x,y,r,c_)
        if r > k + 1e-3:
            continue
        if x*x + y*y >= k*k+ 1e-3:
            continue
        circs.append(c_)
        
    for c_ in pairs:
        pp = np.array([i*b1 + j*b2 for i,j in c_])
    
        x,y = np.average(pp,axis = 0)
        r = np.linalg.norm(pp[1] - pp[0])/2
        c_ = circ(x,y,r,c)
        if c_.r > k + 1e-3:
            continue
        if c_.x*c_.x + c_.y*c_.y > k*k + 1e-3:
            continue
        circs.append(c_)
    circs = np.array(circs)
    return circs

class system:
    
    
    def __init__(self, p = 380e-9, orientation = True, kmax = 2, NA = 0.9, use_wg = True):
        self.p = p
        a1 = np.array([1,0])*p
        
        self.orientation = orientation
        self.a1 = a1 if orientation else R(np.pi/2) @ a1
        self.a2 = R(np.pi/3) @ self.a1
        
        b1,b2 = np.linalg.inv([self.a1,self.a2]).T*2*np.pi
        self.b1 = b1
        self.b2 = b2
        self.b = np.linalg.norm(b1)

        self.wg_params = np.array([-5.06887738e-09,  7.47085964e-01, -4.76509938e+05]) # Parameters from t = 70nm
        #self.wg_params = np.array([-5.06887738e-09,  7.47085964e-01, 0])
        self.poly_ord = len(self.wg_params) - 1

        self.kmax = kmax        
        self.points = get_idx(kmax)
        self.NA = NA

        self.use_wg = use_wg

        circs = calc_HSPs(self.points, self.b1/self.b, self.b2/self.b, kmax)
        self.circles = circs

        rads_wg = np.array([self.wg(c.r*self.b)/self.b for c in circs])
        self.runiq_wg = np.unique(rads_wg)
        
        rads_ela = np.array([c.r for c in circs])
        self.runiq_ela = np.unique(rads_ela)
        self.labs = gen_labels(self.runiq_ela)

        self.k = self.runiq[3]

        px = 1/plt.rcParams['figure.dpi']

        self.fig_data, self.ax_data = plt.subplots(1, figsize = (500*px,500*px), layout = 'constrained')
        self.plot_LC(self.k*self.b)
    
        self.fig_ela, self.ax_ela = plt.subplots(1, figsize = (250*px, 300*px), layout = 'constrained')
        self.ela = self.build_ELA(self.k)[0]

        self.fig_rs, self.ax_rs = plt.subplots(1,figsize = (200*px,200*px),layout = 'constrained')
        self.plot_rs()
        self.fig_ks, self.ax_ks = plt.subplots(1,figsize = (200*px,200*px),layout = 'constrained')    
        self.plot_ks()
        
        self.l1 = 600e-9
        self.l2 = 460e-9
        
        self.fig_disp, self.ax_disp = plt.subplots(1,figsize = (300*px,450*px),layout = 'constrained')
        
        self.disp_lines = []
        self.disp_k = self.plot_dispersion_angles()
        
        
    @property
    def runiq(self):
        if self.use_wg:
            return self.runiq_wg
        else:
            return self.runiq_ela

    def hsp_E(self, hsp):
        for i,l in enumerate(self.labs):
            if l == hsp:
                return self.runiq[i]
        else:
            print("HSP {} not found".format(hsp))
            print("Valid optons : ", self.labs)
            return None

    def orient(self, orientation):
        
        if orientation:
            # Sets orientation along of a1 along x
            a1 = np.array([1,0])*self.p
            self.a1 = a1
            self.a2 = R(np.pi/3) @ a1
            b1,b2 = np.linalg.inv([self.a1,self.a2]).T*2*np.pi
            self.b1 = b1
            self.b2 = b2
            self.orientation = orientation
           
        else:
            # Sets orientation of a1 along y
            self.orientation = orientation
            a1 = np.array([0,1])*self.p
            self.a1 = a1
            self.a2 = R(np.pi/3) @ a1
            b1,b2 = np.linalg.inv([self.a1,self.a2]).T*2*np.pi
            self.b1 = b1
            self.b2 = b2
           

    def S(self,m,n):
        # Structure factor of Honeycomb lattice for DO m,n
        d = (-self.a1 + self.a2)/3*(2*np.pi)
        return (1 + np.cos(np.dot(m*self.b1 + n*self.b2, d)))/2

    def wg(self, k = None):
        if k is None:
            k = self.k
        N = self.poly_ord
        Y = np.array([p*k**(N - j) for j,p in enumerate(self.wg_params)])
        Y = np.sum(Y, axis = 0)
        return Y
    
    def wginv(self, kpar):
        a,b,c = self.wg_params
        c = c - kpar
        Disc = b*b - 4 * a * c
        x1 = (-b + np.sqrt(Disc))/(2*a)
        return x1

    def plot_LC(self, k, plot_kw = {}):
        
        for i,j in self.points:
            if (i == 0) and (j == 0):
                continue
            p = i*self.b1 + j*self.b2
            d = np.linalg.norm(p)
            if d > 3*k:
                continue
            
            pp = p/k
            rad = self.wginv(k)/k if self.use_wg else 1
            
            circle = plt.Circle(pp, rad, fc = 'None', ec = 'k', lw = 2)
            self.ax_data.add_patch(circle)
        
        fov = plt.Circle((0,0), self.NA, fc = 'None', ec = 'r',zorder = -10)
        self.ax_data.add_patch(fov)
        self.ax_data.set_xlim([-self.NA,self.NA])
        self.ax_data.set_ylim([-self.NA,self.NA])
        
        self.ax_data.set_xlabel("kx/|k|")
        self.ax_data.set_ylabel("ky/|k|")
        
        self.ax_data.set_aspect('equal')


    def build_ELA(self, k, N = 100, Nord = 10):
        G = np.array([0,0])
        M = self.b1/2
        K = (self.b1 - self.b2)/3
        b = self.b
        KK = np.linspace(G,M,N)
        KK = np.append(KK,np.linspace(M,K,N)[1:-1], axis = 0)
        KK = np.append(KK,np.linspace(K,G,N), axis = 0)
    
        c = 'C0' if self.use_wg else 'C1'
    
        for i in range(-Nord,Nord + 1):
            for j in range(-Nord,Nord + 1):
                K_ = np.linalg.norm(KK + i*self.b1 + j*self.b2, axis = 1)
                K_ = K_[K_ < 3*self.b]
                if calc_mn(i, j) > Nord:
                    continue
                if self.use_wg:
                    E = self.wg(K_)
                else:
                    E = K_
                
                self.ax_ela.plot(E/b, c = c)
    
        self.ax_ela.set_xlim([0,3*N - 2])
        self.ax_ela.set_ylim([0,np.max(self.runiq)])    
        
    
        xt = [0,N-1,2*N-2,3*N-2]
        self.ax_ela.set_xticks(xt,["G","M", "K", "G"])
        kl = self.ax_ela.plot([0,3*N - 2],[k,k],'k--')
        self.ax_ela.set_ylabel("|k|/|b|")
        return kl
        
    def plot_rs(self):
        self.ax_rs.cla()
        lim = 1.5
        r = 0.1
        D = (2*self.a2 - self.a1)/3/self.p
        for i in range(-5,6):
            for j in range(-5,6):
                if calc_mn(i,j) > 9:
                    continue
                p = (i*self.a1 + j*self.a2)/self.p
                p1 = plt.Circle(p, r, fc = 'C0')
                p2 = plt.Circle(p + D, r, fc = 'C1')
                self.ax_rs.add_patch(p1)
                self.ax_rs.add_patch(p2)

        self.ax_rs.set_aspect('equal')
        self.ax_rs.set_xlim([-lim,lim])
        self.ax_rs.set_ylim([-lim,lim])
        self.ax_rs.set_xlabel("x [p]")
        self.ax_rs.set_ylabel("y [p]")
                
    def plot_ks(self):
        self.ax_ks.cla()
        lim = 1.2
        B = []
        for i in range(-5,6):
            for j in range(-5,6):
                if calc_mn(i,j) > 12:
                    continue
                b = i*self.b1 + j*self.b2
                B.append(b/self.b)
                
        vor = Voronoi(B)
        voronoi_plot_2d(vor, self.ax_ks, show_vertices = False, show_points = True)
        
        self.ax_ks.set_aspect('equal')
        self.ax_ks.set_xlim([-lim,lim])
        self.ax_ks.set_ylim([-lim,lim])
        
        self.ax_ks.set_xlabel("kx [b]")
        self.ax_ks.set_ylabel("ky [b]")
        
    def plot_dispersion_angles(self):
        bb = self.b*2
        Nk = 100
        kk = np.linspace(-bb,bb,Nk)
        
        KK = np.zeros((Nk,2))
        
        KK[:,1] = kk
        
        for i,j in self.points:
            if calc_mn(i, j) > 10:
                continue
            K = np.linalg.norm(i*self.b1 + j*self.b2 - KK, axis = 1)
            if self.use_wg:
                K_ = self.wg(K)
            else:
                K_ = K
            idx = np.where(K_ > 0)
            self.ax_disp.plot(kk[idx]/K_[idx],K_[idx]*scale, c = 'C1')[0]
        
        self.ax_disp.set_xlim([-self.NA,self.NA])
        e1 = lam2eV(self.l1)
        e2 = lam2eV(self.l2)
        self.ax_disp.set_ylim([e1,e2])
        Nt = 5
        Yt = np.linspace(self.l1,self.l2,Nt)
        Yt = lam2eV(Yt)
        Yl = np.linspace(self.l1,self.l2,Nt)*1e9
        Yl = [int(y) for y in Yl]
        self.ax_disp.set_yticks(Yt,Yl)
        
        ee = self.k*scale*self.b
        
        xlab = "ky/|k|"
        
        self.ax_disp.set_xlabel(xlab)
        self.ax_disp.set_ylabel("wavelength [nm]")

        return self.ax_disp.plot([-self.NA,self.NA], [ee,ee],'k--')[0]
    
    
    def update_p(self, new_p):
        self.p = new_p
        
        a1 = np.array([1,0])*new_p
        self.a1 = a1 if self.orientation else R(np.pi/2) @ a1
        self.a2 = R(np.pi/3) @ self.a1
        
        b1,b2 = np.linalg.inv([self.a1,self.a2]).T*2*np.pi
        self.b1 = b1
        self.b2 = b2
        self.b = np.linalg.norm(b1)
        
        rads_wg = np.array([self.wg(c.r*self.b)/self.b for c in self.circles])
        self.runiq_wg = np.unique(rads_wg)
        
        rads_ela = np.array([c.r for c in self.circles])
        self.runiq_ela = np.unique(rads_ela)
        self.disp_k = self.plot_dispersion_angles()
"""
Update functions
"""
def update_p(*_):
    b0 = sys.b
    k0 = sys.k
    pp = period.get()/1e9
    sys.update_p(pp)
    
    kb = k0*b0
    winv = sys.wginv(kb)
    winv *= sys.b/b0
    sys.k = sys.wg(winv)/sys.b
    

    for l in sys.ax_disp.lines:
        l.remove()
    
    sys.disp_k = sys.plot_dispersion_angles()
    sys.ax_ela.cla()
    sys.ela = sys.build_ELA(sys.k)[0]
    
    ee = sys.k*sys.b*scale

    sys.disp_k.set_ydata([ee,ee])
    canvas_disp.draw()
    canvas_ela.draw()
    
    k_ = sys.k*sys.b
        
    for p in sys.ax_data.patches:
        p.remove()
        
    sys.plot_LC(k_)
    canvas.draw()

def update_orientation(*_):
    sys.orient(SF.get())
    sys.plot_rs()
    sys.plot_ks()
    canvas_rs.draw()
    canvas_ks.draw()
    canvas_disp.draw()
    
    for p in sys.ax_data.patches:
        p.remove()
    sys.plot_LC(sys.k*sys.b)
    canvas.draw()
    
    for l in sys.ax_disp.lines:
        l.remove()
    sys.disp_k = sys.plot_dispersion_angles()
    ee = sys.k*sys.b*scale

    sys.disp_k.set_ydata([ee,ee])
    canvas_disp.draw()
    
def update_k(*_):
    
    k_ = kk.get()*sys.b
    sys.k = k_/sys.b
        
    for p in sys.ax_data.patches:
        p.remove()
        
    sys.plot_LC(k_)
    sys.ela.set_ydata([kk.get(),kk.get()])
    sys.disp_k.set_ydata([k_*scale,k_*scale])
    canvas_ela.draw()
    canvas_disp.draw()
    canvas.draw()
  


def import_file():
    filename = tk.filedialog.askopenfilename(title = "Select a file", filetypes = [("images","*.png"),("Numpy","*.npz")])
    
    if filename:
        A = np.zeros(1)
        if ".npz" in filename:
            data = np.load(filename)
            if "A" in data.keys():
                A = data["A"]
            else:
                A = data[list(data.keys())[0]]

            
        elif ".png" in filename:
            A = get_img_np(filename, 710)

        else:
            return None
        
        N,M = A.shape
        X = np.linspace(-sys.NA,sys.NA,N)
        Y = np.linspace(-sys.NA,sys.NA,M)
        sys.ax_data.cla()
        sys.ax_data.pcolormesh(X,Y,A,zorder = -10)
        sys.ax_data.set_xlabel("kx/|k|")
        sys.ax_data.set_ylabel("ky/|k|")
        canvas.draw()
        

def import_file_dispersion():
    filename = tk.filedialog.askopenfilename(title = "Select a file", filetypes = [("images","*.png")])
    
    if filename:

        A = crop_rectangle_data(filename)

        
        N,M = A.shape
        X = np.linspace(-sys.NA,sys.NA,M)
        Y = np.linspace(lam2eV(sys.l2),lam2eV(sys.l1),N)
        sys.ax_disp.cla()
        sys.ax_disp.pcolormesh(X,Y,A,zorder = -10)
        sys.disp_k = sys.plot_dispersion_angles()
        ee = sys.k*sys.b*scale

        sys.disp_k.set_ydata([ee,ee])
        canvas_disp.draw()
        
"""
GUI and script
"""

sys = system()

root = tk.Tk()
root['bg'] = 'white'
root.geometry("1250x550")
root.title("Kristian's Lightcone analyzer")

root.config(cursor="")

# Frame for options and ELA dispersions
frame_opts = tk.Frame(root, width = 400, height = 300,bg = 'white')
frame_opts.grid(column = 0, row = 0, padx = 5, pady = 5, sticky= 'w'+'e'+'n'+'s')

# Frame for lightcones
frame = tk.Frame(root, width = 500, height = 500,bg = 'white')
frame.grid(row = 0, column = 1, padx = 5, pady = 5, rowspan = 2,sticky= 'w'+'e'+'n'+'s')

# Frame for ks and rs pictures
frame_lattices = tk.Frame(root, width = 400, height = 200,bg = 'white')
frame_lattices.grid(column = 0, row = 1,padx = 5, pady = 5, sticky= 'w'+'e'+'n'+'s')

# Frame for everything dispersion
frame_disp_0 = tk.Frame(root, width = 300, height = 500,bg = 'white')
frame_disp_0.grid(column = 2, row = 0, rowspan = 2,padx = 5, pady = 5, sticky = 'wens')

# Dispersion options
frame_disp_opts = tk.Frame(frame_disp_0, height = 50, width = 300,bg = 'white')
frame_disp_opts.grid(row = 0, column = 0)

# Frame for dispersion plot
frame_disp = tk.Frame(frame_disp_0, height = 450, width = 300,bg = 'white')
frame_disp.grid(row = 1, column = 0)


# k, passed to plots
kk = tk.DoubleVar()
kk.set(sys.runiq[3])

# period, passed to sys
period = tk.DoubleVar()
period.set(sys.p*1e9)


rrow = 0
# Import data for lightcone
import_button = tk.Button(frame_opts, text= "Load data", command = import_file)
import_button.grid(row = rrow, column = 0, sticky= 'wens', padx = 5, pady = 5)
rrow += 1


import_disp_button = tk.Button(frame_disp_opts, text= "Load dispersion", command = import_file_dispersion)
import_disp_button.grid(row = 0, column = 0, sticky= 'wens', padx = 5, pady = 5, rowspan = 2)


kscale = tk.Scale(frame_opts, variable = kk,
                  from_ = 0, to = np.max(sys.runiq),
                  orient = tk.HORIZONTAL,
                  resolution = 0.01,
                  command = update_k)

kscale.grid(row = rrow, column = 0)
rrow += 1

plab = tk.Label(frame_disp_opts, text = "period [nm]")
plab.grid(column= 1, row = 0)
pscale = tk.Scale(frame_disp_opts, variable = period,
                  from_ = 300, to = 500,
                  orient = tk.HORIZONTAL,
                  resolution = 1,
                  command = update_p
                  )
pscale.grid(row = 1, column = 1)

SF = tk.BooleanVar()
Sbox = tk.Checkbutton(
        frame_opts, text = "Orientation",
        variable = SF,
        onvalue = True,
        offvalue = False,
        command = update_orientation
    )

Sbox.grid(row = rrow, column = 0, columnspan = 4, sticky = 'wens')
SF.set(True)
rrow += 1
def func(l):
    r = sys.hsp_E(l)
    kk.set(r)
    update_k(kk)

cols = {"G" : 0, "M" : 1, "K" : 2, "T" : 3}
rows = {
        "G" : 0,
        "M" : 0,
        "K" : 0,
        "T" : 0
        }

button_frame = tk.Frame(frame_opts, width = 150, height = 200, bg = 'grey')
button_frame.grid(row = rrow, column = 0)
rrow += 1
for l in sys.labs:
    button = tk.Button(button_frame, text = "{}".format(l), command = lambda l=l: func(l))
    button.grid(column = cols[l[0]], row = rows[l[0]])#,sticky='w'+'e'+'n'+'s')
    rows[l[0]] += 1
    
canvas_ela = FigureCanvasTkAgg(sys.fig_ela, master = frame_opts)
canvas_ela.get_tk_widget().grid(column = 4, row = 0,columnspan = 4, rowspan = rrow,sticky = 'wens', padx = 10, pady = 5)

canvas = FigureCanvasTkAgg(sys.fig_data, master = frame)
canvas.get_tk_widget().grid(column = 0, row = 0)


canvas_rs = FigureCanvasTkAgg(sys.fig_rs, master = frame_lattices)
canvas_rs.get_tk_widget().grid(column = 0, row = 0, sticky = 'wens')
canvas_ks = FigureCanvasTkAgg(sys.fig_ks, master = frame_lattices)
canvas_ks.get_tk_widget().grid(column = 1, row = 0, sticky = 'wens')

canvas_disp = FigureCanvasTkAgg(sys.fig_disp, master = frame_disp)
canvas_disp.get_tk_widget().grid(column = 0, row = 0, sticky = 'wens')



root.mainloop()