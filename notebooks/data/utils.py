
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Ellipse

def plot( x, y, ax=None, mask=None, nbins=100, log=True, size=5, cmap="Blues",
          cmap_min=None, alpha=1.0, xlim=None, ylim=None, vlim=None,
          xlab="", ylab="", aspect=None, show=True ):
    
    if xlim is None:
        xlim = [ x.min(), x.max() ]

    if ylim is None:
        ylim = [ y.min(), y.max() ]

    if mask is not None:
        # note, applying the mask *after* computing xlimand ylim
        x = x[mask]
        y = y[mask]
        
    xbins = np.linspace( xlim[0], xlim[1], nbins)
    ybins = np.linspace( ylim[0], ylim[1], nbins)
    
    h, _, _ = np.histogram2d( x, y, bins=(xbins,ybins) )
    h = h.transpose()  # See note at end of numpy.histogram2d() documentation
        
    if log:
        sel = h>0
        h[sel] = np.log10(h[sel])

    if vlim is None:
        vlim = [ h.min(), h.max() ]

    if cmap_min is not None:
        cmap = get_cmap(cmap)
        #cmap.set_under( 'white', alpha=0 )
        cmap.set_under( cmap_min )
        eps = 1e-10  # a small float
        vlim[0] = vlim[0] + eps  # a trick to set the min value to white

    kwargs = { 'origin':'lower', 'cmap':cmap, 'extent':xlim+ylim, 'vmin':vlim[0], 'vmax':vlim[1], 'aspect':aspect, 'alpha':alpha }

    if ax is not None:
        ax.imshow( h, **kwargs )    
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    else:
        plt.imshow( h, **kwargs )    
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        
    if show:
        plt.show()
    

def image( gal, mask=None, nbins=100, log=True, size=5, cmap="afmhot", lim=(-20,20), vlim=(0.0,4.0) ):
    
    kwargs = { 'mask':mask, 'nbins':nbins, 'log':log, 'size':size, 'cmap':cmap, 'show':False,
               'xlim':lim, 'ylim':lim, 'vlim':vlim, 'xlab':"x (kpc)", 'ylab':"y (kpc)" }
    
    fig, ax = plt.subplots( 1, 2, figsize=(size*2,size) )        
    
    plot( gal.x, gal.y, ax[0], **kwargs )
    
    kwargs['ylab'] = "z (kpc)"
    plot( gal.x, gal.z, ax[1], **kwargs )
    
    plt.show()


def corner( X, mask=None, split=False, nbins=100, log=True, size=3, cmap="Blues", cmap_split="Reds", cmap_min='white', aspect='auto',
           labs=None, stars=None, stars_kwargs={'marker':'*','c':'white','s':150,'edgecolors':'black'},
           gmm=None):
    
    kwargs = { 'mask':mask, 'nbins':nbins, 'log':log, 'size':size, 'cmap':cmap, 'cmap_min':cmap_min, 'aspect':aspect, 'show':False }
    
    if split:
        split_kwargs = kwargs.copy()
        split_kwargs['mask'] = ~kwargs['mask']
        split_kwargs['cmap'] = cmap_split

    nfeatures = X.shape[1]
    nn = nfeatures - 1
    
    if labs is None:
        labs = [""]*nfeatures
    
    fig, axs = plt.subplots( nn, nn, figsize=(size*nn,size*nn) )        
    
    for i in range(1,nfeatures):
        for j in range(nn):
            
            if nfeatures > 2:
                ax = axs[i-1,j]
            else:
                ax = axs
            
            if i<=j:
                ax.axis('off')
            else:
                xlab = "" if i<nn else labs[j]
                ylab = "" if j>0 else labs[i]
                plot( X[:,j], X[:,i], ax, xlab=xlab, ylab=ylab, **kwargs )
              
                if split:
                    plot( X[:,j], X[:,i], ax, xlab=xlab, ylab=ylab, alpha=0.5, **split_kwargs )
                    
                if stars is not None:
                    ax.scatter( stars[:,j], stars[:,i], **stars_kwargs )

                if gmm is not None:
                    # Draw ellipses corresponding to the Gaussian Mixture Model
                    means = gmm.means_[:,(j,i)]
                    covs = np.zeros( (means.shape[0],2,2) )
                    for k in range(covs.shape[0]):
                        covs[k,0,0] = gmm.covariances_[k,j,j]  # vars_j
                        covs[k,1,1] = gmm.covariances_[k,i,i]  # vars_i
                        covs[k,0,1] = gmm.covariances_[k,i,j]  # covs_ij
                        covs[k,1,0] = covs[k,0,1]
                    plot_ellipse(means, covs, ax)

    plt.show()

def plot_ellipse(means, covs, ax):
    colors = ['navy', 'turquoise', 'darkorange','red','green','blue','black','magenta','yellow','cyan']
    for mean, cov, color in zip(means, covs, colors):
        v, w = np.linalg.eigh(cov)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        #ax.set_aspect('equal', 'datalim')


def show_random(images, labels, predictions=None, nrow=5, ncol=5, size=2):
    ''' Produces a plot of some random images and their associated labels '''

    fig, ax = plt.subplots( nrow, ncol, figsize=(size*ncol,size*nrow) )
    indexes = np.random.choice( range(images.shape[0]), size=nrow*ncol, replace=False ).tolist()

    for i in range(nrow):
        for j in range(ncol):
            index = indexes.pop(0)
            im = images[ index, : ]
            lab = "{:d}".format( labels[ index ] )

            if predictions is not None:
                lab = "{}\n{:1.2f}".format(lab, predictions[index])

            ax[i,j].imshow(im)

            # Text label
            ax[i,j].text(0.02, 0.035, lab, transform=ax[i,j].transAxes, fontsize=14, color='white',
                        bbox=dict(facecolor='grey', edgecolor='none', alpha=0.7, boxstyle='square,pad=0.3') )

            # Turn axes off, and turn axes annotation off:
            #ax[i,j].axis('off')
            ax[i,j].set_axis_off()

            # Reduce whitespace between subplots:
            eps = 0.05
            plt.subplots_adjust(left=0., bottom=0, right=1, top=1, wspace=eps, hspace=eps)

    plt.show()
    plt.close()
