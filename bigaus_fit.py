import matplotlib.pyplot as plt
from numpy import *
import matplotlib.path as mpltPath
from matplotlib.backend_bases import MouseButton
from iminuit import cost, Minuit
from scipy.stats import norm
from uncertainties import ufloat, correlated_values
import traceback

def BiGausPdf(x,A1,m1,s1,A2,m2,s2):
    return A1*norm.pdf(x,m1,s1) + A2*norm.pdf(x,m2,s2)

def BiGausCdf(x,A1,m1,s1,A2,m2,s2):
    return A1*norm.cdf(x,m1,s1) + A2*norm.cdf(x,m2,s2)

class BiGausFitter:
    '''
    '''
    def __init__(
        self,counts,edges,
        par0=None,dofit=False,dohesse=False,
        title=None,figsize=(6,6*0.75),axis_=None
    ):
        '''
    Constructor
        '''
        self.x_ = array([])
        self.y_ = array([])
        self.counts_ = array(counts)
        self.edges_  = array(edges)
        
        self.minuit_, self.cost_ = None, None
        
        if axis_ is not None:
            self.ax_ = axis_
            self.fig_ = self.ax_.get_figure()
            #plt.sca(self.ax_)
        else:
            self.fig_,self.ax_ = plt.subplots(1,1,figsize=figsize)
        if title is not None:
            self.ax_.text(
                0.5,1.02,title,fontsize=16,
                ha='center',transform=self.ax_.transAxes
            )
        self.ax_.stairs(self.counts_,self.edges_,color='k')
        bb = self.edges_[:-1] + 0.5*(self.edges_[1]-self.edges_[0])
        cc = self.counts_
        self.ax_.errorbar(bb,cc,sqrt(cc),marker='none',ls='none',color='k')
        self.fig_.canvas.draw()
        self.fig_.canvas.flush_events()
        if par0 is None:
            self.from_clicks(dofit=dofit,dohesse=dohesse)
        else:
            self.p0_ = par0
            A1,m1,s1,A2,m2,s2 = self.p0_
            xp = linspace(self.edges_[0], self.edges_[-1],10000)
            self.ax_.plot(xp, A1*norm.pdf(xp,m1,s1) + A2*norm.pdf(xp,m2,s2), ls='--', label='Initial Guesses')
            if dofit:
                self.do_fit(dohesse=dohesse)
            self.ax_.legend(fontsize=12)

        
    def counts(self):
        return self.counts_
    
    def edges(self):
        return self.edges_
    
    def par0(self):
        '''
    Returns initial parameter guesses from mouse clicks.
        '''
        return self.p0_
    
    def minuit(self):
        '''
    Returns Minuit fitter
        '''
        return self.minuit_
    
    def ufpars(self):
        '''
        Returns fitted parameters as ufloats with correlated uncertainties.
        '''
        return correlated_values(self.minuit().values, self.minuit().covariance)
    
    def FOM(self):
        '''
        returns figure of merit defined by:
            FOM = |m1 - m2| / [2.355*(s1+s2)]
        includes uncertainty propagation on the fitted parameters m1,m2,s1,s2.
        '''
        pfit = self.ufpars()
        return abs(pfit[1]-pfit[4])/(2.355*(pfit[2]+pfit[5]))
    
    def peak_counts(self):
        '''
        returns integrated counts in each fitted peak, with uncertainty propagation.
        returned as (<leftmost peak counts>, <rightmost peak counts>)
        '''
        pfit = self.ufpars()
        binwidth = self.edges()[1]-self.edges()[0]
        if pfit[1].nominal_value < pfit[4].nominal_value:
            return (pfit[0]/binwidth, pfit[3]/binwidth)
        else:
            return (pfit[3]/binwidth, pfit[1]/binwidth)
    
    def do_fit(self, dohesse = True):
        self.cost_ = cost.ExtendedBinnedNLL(
            self.counts_, self.edges_, BiGausCdf
        )
        cc = self.counts_ > 0
        bw = self.edges_[1] - self.edges_[0]
        bb = self.edges_[:-1] + 0.5*bw
        self.cost_ = cost.LeastSquares(
            bb[cc], self.counts_[cc], sqrt(self.counts_[cc]),
            BiGausPdf
        )
        self.minuit_ = Minuit(
            self.cost_,self.p0_[0],self.p0_[1],self.p0_[2],self.p0_[3],self.p0_[4],self.p0_[5])
        self.minuit_.migrad()
        if dohesse:
            self.minuit_.hesse()
        
        self.plot_fit()
        
    def plot_fit(self,includelegend=True,newAx=None):
        xp = linspace(self.edges_[0], self.edges_[-1], 10000)
        A1,m1,s1,A2,m2,s2 = self.minuit_.values
        validlabel = 'valid' if self.minuit_.valid else 'invalid'
        theLabel = 'Minuit Fit (%s)'%validlabel
        if self.minuit_.valid:
            theLabel += '\nA1=%3.3g\nm1=%3.3g\ns1=%3.3g\nA2=%3.3g\nm1=%3.3g\ns1=%3.3g' % (
                A1,m1,s1,A2,m2,s2
            )
            theLabel += '\nFOM={:.1fS}'.format(self.FOM())
            theLabel += '\nN1={:.1fS}\nN2={:.1fS}'.format(
                self.peak_counts()[0], self.peak_counts()[1]
            )
        thisAx = self.ax_ if newAx is None else newAx
        thisAx.plot(
            xp,
            BiGausPdf(xp,A1,m1,s1,A2,m2,s2)
            , ls = '-', color = 'b'
            , label = theLabel
        )
        thisAx.plot(
            xp,
            BiGausPdf(xp,A1,m1,s1,0,m2,s2)
            , ls = ':', color = 'b'
        )
        thisAx.plot(
            xp,
            BiGausPdf(xp,0,m1,s1,A2,m2,s2)
            , ls = ':', color = 'b'
        )
        if includelegend:
            thisAx.legend(fontsize=10)

    def from_clicks(self,color='fuchsia',dofit=False,dohesse=False):
        self.x_ = array([])
        self.y_ = array([])
        self.p0_ = ones(6,float)
        self.pplt_ = []
        def on_click(event):
            if event.inaxes is not self.ax_:
                return
            if event.button is MouseButton.LEFT or event.button is MouseButton.RIGHT:
                self.x_ = append(self.x_, event.xdata)
                self.y_ = append(self.y_, event.ydata)
            else:
                return
            self.pplt_.append (
                self.ax_.plot(self.x_,self.y_,linestyle='none',marker='o',color=color)
            )
            if len(self.x_) == 1:
                self.line1_=self.ax_.axhline(self.y_[0]/2, ls = '--', color = 'r')
            if len(self.x_) == 4:
                self.line2_=self.ax_.axhline(self.y_[3]/2, ls = '--', color = 'b')
            if len(self.x_) == 6:
                xp = linspace(self.edges_[0], self.edges_[-1],10000)
                m1,m2 = self.x_[0], self.x_[3]
                s1,s2 = abs(self.x_[2]-self.x_[1])/2.355, abs(self.x_[4]-self.x_[5])/2.355
                A1,A2 = self.y_[0]*s1*sqrt(2*pi), self.y_[3]*s2*sqrt(2*pi)
                self.p0_ = array([A1,m1,s1,A2,m2,s2],float)
                self.ax_.plot(
                    xp, 
                    A1*norm.pdf(xp,m1,s1) + A2*norm.pdf(xp,m2,s2), 
                    ls='--', label='Initial Guess', color = 'r'
                )
                self.ax_.legend()
                self.line1_.remove()
                self.line2_.remove()
                [ pp[0].remove() for pp in self.pplt_ ]
                if dofit:
                    self.do_fit(dohesse=dohesse)
                plt.disconnect(binding_id)
            self.fig_.canvas.draw()
                    
        binding_id = plt.connect('button_press_event', on_click)